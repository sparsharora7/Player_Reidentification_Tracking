import cv2
from ultralytics import YOLO
from src.reid_tracker import PlayerTracker
from src.utils import cosine_similarity

# ✅ Load the fine-tuned YOLOv11 model from 'best.pt'
model = YOLO("model/best.pt")

# Load video
cap = cv2.VideoCapture("input/15sec_input_720p.mp4")
tracker = PlayerTracker()

# Prepare output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output/reid_output.mp4", fourcc, 30.0, (1280, 720))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []

    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        detections.append(((x1, y1, x2, y2), crop))

    tracked = tracker.update(detections)

    for track_id, bbox in tracked.items():
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
