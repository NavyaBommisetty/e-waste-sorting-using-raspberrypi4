import cv2
import torch

# ----------------------------
# Load custom YOLOv8 model
# ----------------------------
# Replace 'best.pt' with your trained Roboflow model
model = torch.hub.load('ultralytics/yolov8', 'custom', path='best.pt', force_reload=True)

# ----------------------------
# Initialize Raspberry Pi Camera
# ----------------------------
cap = cv2.VideoCapture(0)  # Use PiCamera or USB camera

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # YOLO inference
        results = model(frame)

        # Draw bounding boxes and labels
        for r in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = r.tolist()
            label = model.names[int(cls)]
            confidence = round(conf, 2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} {confidence}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("E-Waste Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Exiting...")

finally:
    cap.release()
    cv2.destroyAllWindows()

