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
        self.fps = 12
        self.no_person_count = 0
        self.person_detected = 0
        self.cap = None
        self.camera_index = 0

        self.settings = self.load_settings()

    def load_settings(self):
        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
            print("Settings loaded successfully:", settings)
            return settings
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading settings: {e}")
            return {}

    def apply_camera_settings(self):
        if 'brightness' in self.settings:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.settings['brightness'])
        if 'contrast' in self.settings:
            self.cap.set(cv2.CAP_PROP_CONTRAST, self.settings['contrast'])
        if 'saturation' in self.settings:
            self.cap.set(cv2.CAP_PROP_SATURATION, self.settings['saturation'])

    def capture_frames(self):
        model = YOLO('./models/model.pt')
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Apply camera settings from JSON
        self.apply_camera_settings()

        while True:
            success, frame = self.cap.read()
            if success:
                results = model.track(frame, classes=0, conf=0.6, imgsz=480, verbose=False)
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
                    self.video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), self.fps, (640, 480))
                    print("Started saving video.")

                if self.video_saving:
                    self.frame_buffer.append(frame)
                    self.video_writer.write(frame)

                    if not person_detected:
                        self.no_person_count += 1

                    if len(self.frame_buffer) >= 50 or self.no_person_count >= 5:
                        image_path = f"captures/image_{timestamp}.jpg"
                        cv2.imwrite(image_path, self.frame_buffer[int(len(self.frame_buffer) / 2)])
                        self.video_saving = False
                        self.video_writer.release()
                        self.video_writer = None
                        self.frame_buffer.clear()
                        self.no_person_count = 0
                        print("Finished saving video.")

                        # Send POST request to Express.js server
                        try:
                            response = requests.post(
                                "http://localhost:3000/images/upload",
                                json={"filePath": home + image_path, "totalEntity": self.person_detected},
                                headers={"Content-Type": "application/json"}
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
            else:
                print("Error reading frame")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.video_writer.release()
                break
            time.sleep(0.1)

        self.cap.release()

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
            brightness=self.settings['brightness'],
            contrast=self.settings['contrast'],
            saturation=self.settings['saturation']
        )

    def SetCameraSettings(self, request, context):
        self.settings = {
            'brightness': request.brightness if request.brightness >= 0 else self.settings['brightness'],
            'contrast': request.contrast if request.contrast >= 0 else self.settings['contrast'],
            'saturation': request.saturation if request.saturation >= 0 else self.settings['saturation']
        }

        self.apply_camera_settings()

        with open("config/camera_settings.json", 'w') as f:
            json.dump(self.settings, f)
        print("Camera settings updated:", self.settings)
        return pb2.Empty()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_MonitoringServicer_to_server(CCTVService(), server)
    server.add_insecure_port('[::]:50051')
    print("Server started on port 50051.")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
