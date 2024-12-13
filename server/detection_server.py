import cv2
import grpc
import json
from concurrent import futures
import threading
import time
import cctv_pb2_grpc as pb2_grpc
import cctv_pb2 as pb2
from ultralytics import YOLO
import collections
import requests
import os

home = os.getcwd() + "/"
settings_file = home + "config/camera_settings.json"
ip_config_file = home + "config/ip_config.json"
grpc_file = home + "config/grpc.json"

class CCTVService(pb2_grpc.MonitoringServicer):
    def __init__(self):
        super().__init__()
        self.frame = None
        self.lock = threading.Lock()  # Lock to handle frame safely across threads
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        self.video_saving = False
        self.video_writer = None
        self.frame_buffer = collections.deque(maxlen=50)
        self.detection_buffer = collections.deque(maxlen=50)
        self.fps = 12
        self.no_person_count = 0
        self.person_detected = 0
        self.cap = None
        self.camera_index = "rtsp://192.168.1.1/live/ch00_1?rtsp_transport=tcp"
        self.is_using_rtsp = str(self.camera_index).startswith("rtsp://")

        self.settings = self.load_settings()
        self.ip_config = self.load_ip_config()

    def load_settings(self):
        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
            print("Settings loaded successfully:", settings)
            return settings
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading settings: {e}")
            return {}

    def load_ip_config(self):
        if not os.path.exists(ip_config_file):
            print(f"Error: IP config file '{ip_config_file}' not found.")
            return {"host": "localhost", "port": 3000}

        try:
            with open(ip_config_file, 'r') as f:
                ip_config = json.load(f)
                if 'host' not in ip_config or 'port' not in ip_config:
                    raise ValueError("Missing 'host' or 'port' in IP config file.")
                print("IP config loaded successfully:", ip_config)
                return ip_config
        except json.JSONDecodeError as e:
            print(f"Error parsing IP config file '{ip_config_file}': {e}")
            return {"host": "localhost", "port": 3000}
        except ValueError as e:
            print(f"Error: {e}")
            return {"host": "localhost", "port": 3000}

    def apply_camera_settings(self):
        if not self.is_using_rtsp:
            if 'brightness' in self.settings:
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.settings['brightness'])
            if 'contrast' in self.settings:
                self.cap.set(cv2.CAP_PROP_CONTRAST, self.settings['contrast'])
            if 'saturation' in self.settings:
                self.cap.set(cv2.CAP_PROP_SATURATION, self.settings['saturation'])

    def map_value(self,value, from_min, from_max, to_min, to_max):
        return (value - from_min) / (from_max - from_min) * (to_max - to_min) + to_min

    def adjust_image(self, image, alpha=1.0, beta=0, saturation_scale=1.0):
        adjusted = cv2.addWeighted(image, alpha, image, 0, beta)

        hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, saturation_scale)
        s = cv2.min(s, 255)

        hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def capture_frames(self):
        model = YOLO('./models/model.pt')
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_FFMPEG)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.apply_camera_settings()

        while True:
            if self.is_using_rtsp:
                for _ in range(5):
                    self.cap.read()
            success, frame = self.cap.read()
            if success:
                if self.is_using_rtsp:
                    frame = self.adjust_image(
                        frame,
                        self.map_value(self.settings.get("contrast"), 0, 255, 0.0, 3.0),
                        self.map_value(self.settings.get("brightness"), 0, 255, -255, 255),
                        self.map_value(self.settings.get("saturation"), 0, 255, 0.0, 2.0))

                results = model.track(frame, classes=0, conf=0.6, imgsz=320, verbose=False)
                person_detected = len(results[0].boxes) > 0
                self.person_detected = len(results[0].boxes)

                for i, box in enumerate(results[0].boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]

                    label = f"id: {i}  {class_name} {confidence:.2f}"
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_width, y1), (0, 0, 255), -1)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if person_detected and not self.video_saving:
                    self.video_saving = True
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    video_path = f"/captures/video_{timestamp}.avi"
                    self.video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), self.fps, (320, 240))
                    print("Started saving video.")

                if self.video_saving:
                    self.frame_buffer.append(frame)
                    self.detection_buffer.append(self.person_detected)
                    self.video_writer.write(frame)

                    if not person_detected:
                        self.no_person_count += 1

                    if len(self.frame_buffer) >= 50 or self.no_person_count >= 5:
                        image_path = f"captures/image_{timestamp}.jpg"
                        index = int(len(self.frame_buffer) / 2)
                        image_saved = self.frame_buffer[index]
                        detection_saved = self.detection_buffer[index]
                        cv2.imwrite(image_path, image_saved)
                        self.video_saving = False
                        self.video_writer.release()
                        self.video_writer = None
                        self.frame_buffer.clear()
                        self.detection_buffer.clear()
                        self.no_person_count = 0
                        self.person_detected = 0
                        print("Finished saving video.")

                        # Send POST request to Express.js server
                        if detection_saved > 0:
                            try:
                                with open(image_path, 'rb') as img_file:
                                    response = requests.post(
                                        f"http://{self.ip_config['host']}:{self.ip_config['port']}/images/upload",
                                        files={"image": img_file},
                                        data={"totalEntity": detection_saved}
                                    )
                                    if response.status_code == 201:
                                        print("Image uploaded successfully.")
                                    else:
                                        print(f"Failed to upload image: {response.status_code}, {response.text}")
                            except requests.RequestException as e:
                                print(f"Error uploading image: {e}")

                _, buffer = cv2.imencode('.jpg', frame)
                with self.lock:
                    self.frame = buffer.tobytes()
                # cv2.imshow('frame', frame)
            else:
                print("Error reading frame")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                if self.video_writer:
                    self.video_writer.release()
                break
            time.sleep(0.1)

        self.cap.release()
        # cv2.destroyAllWindows()

    def GetImage(self, request, context):
        with self.lock:
            if self.frame is None:
                print("No frame available")
                return pb2.Image(data=b"")
            return pb2.Image(data=self.frame)

    def GetDetection(self, request, context):
        return pb2.Detection(count=self.person_detected)

    def GetCameraSettings(self, request, context):
        return pb2.CameraSettings(
            brightness=self.settings.get('brightness', 0),
            contrast=self.settings.get('contrast', 0),
            saturation=self.settings.get('saturation', 0)
        )

    def SetCameraSettings(self, request, context):
        self.settings = {
            'brightness': request.brightness if request.brightness >= 0 else self.settings.get('brightness', 0),
            'contrast': request.contrast if request.contrast >= 0 else self.settings.get('contrast', 0),
            'saturation': request.saturation if request.saturation >= 0 else self.settings.get('saturation', 0)
        }

        self.apply_camera_settings()

        with open(settings_file, 'w') as f:
            json.dump(self.settings, f)
        print("Camera settings updated:", self.settings)
        return pb2.Empty()

def serve():
    grpc_config = {}
    with open(grpc_file, 'r') as f:
        grpc_config = json.load(f)
    port = grpc_config.get('port', 50051)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_MonitoringServicer_to_server(CCTVService(), server)
    server.add_insecure_port(f'[::]:{port}')
    print(f"Server started on port {port}.")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
