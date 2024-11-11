import cv2
import grpc
from concurrent import futures
import threading
import time
import cctv_pb2_grpc as pb2_grpc
import cctv_pb2 as pb2
from ultralytics import YOLO

class CCTVService(pb2_grpc.MonitoringServicer):
    def __init__(self):
        super().__init__()
        self.frame = None
        self.lock = threading.Lock()  # Lock to handle frame safely across threads
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def capture_frames(self):
        model = YOLO('./models/model.pt')
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        while True:
            success, frame = cap.read()
            if success:
                results = model.track(frame, classes=0, conf=0.8, imgsz=480, verbose=False)
                cv2.putText(frame, f"Total: {len(results[0].boxes)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
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

                # cv2.imshow("Frame", frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                _, buffer = cv2.imencode('.jpg', frame)
                with self.lock:
                    self.frame = buffer.tobytes()
            else:
                print("Error reading frame")
            time.sleep(0.1)

        cap.release()
        # cv2.destroyAllWindows()

    def GetImage(self, request, context):
        # print("Send frame to client")
        with self.lock:
            if self.frame is None:
                print("No frame available")
                return pb2.Image(data=b"")
            return pb2.Image(data=self.frame)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_MonitoringServicer_to_server(CCTVService(), server)
    server.add_insecure_port('[::]:50051')
    print("Server started on port 50051.")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
