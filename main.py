from ultralytics import YOLO
import cv2
import torch

# ==============================
# โหลดโมเดล
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("⚡ Using device:", device)

model_person = YOLO("./best_person.pt").to(device)    # โมเดลตรวจคน
model_glasses = YOLO("./best_glasses.pt").to(device)  # โมเดลตรวจแว่น

# ==============================
# เปิดกล้อง
# ==============================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ ไม่สามารถเปิดกล้องได้")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame = frame.copy()

    # ==============================
    # รันโมเดลตรวจคน
    # ==============================
    results_person = model_person.predict(frame, conf=0.3, verbose=False)
    for box in results_person[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = results_person[0].names[cls_id]

        # วาดกรอบสีเขียว
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Person/{cls_name} {conf:.2f}",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    # ==============================
    # รันโมเดลตรวจแว่น
    # ==============================
    results_glasses = model_glasses.predict(frame, conf=0.3, verbose=False)
    for box in results_glasses[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = results_glasses[0].names[cls_id]

        # วาดกรอบสีน้ำเงิน
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(annotated_frame, f"Glasses/{cls_name} {conf:.2f}",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2)

    # ==============================
    # แสดงผล
    # ==============================
    cv2.imshow("YOLOv11 Person + Glasses Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
