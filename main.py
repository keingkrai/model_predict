from ultralytics import YOLO
import cv2
import torch
import joblib
import numpy as np
from facenet_pytorch import InceptionResnetV1

# ==============================
# โหลดโมเดล
# ==============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("⚡ Using device:", device)

# YOLO ตรวจ "หน้า" (อาจใช้ yolov8n-face.pt หรือ train เอง)
model_face = YOLO("./yolov8n-face.pt").to(device)

# Face embedding + SVM classifier
embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)
svm_clf = joblib.load("face_svm.pkl")
le = joblib.load("label_encoder.pkl")

# โมเดลตรวจแว่น (ตามที่คุณ train ไว้แล้ว)
model_glasses = YOLO("./best_glasses.pt").to(device)

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
    # ตรวจ "ใบหน้า" ด้วย YOLO
    # ==============================
    results_face = model_face.predict(frame, conf=0.5, verbose=False)
    for box in results_face[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        face = frame[y1:y2, x1:x2]

        # เตรียม face → embedding
        face_resized = cv2.resize(face, (160, 160))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_tensor = torch.tensor(face_rgb/255., dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = embedder(face_tensor).cpu().numpy()

        # Predict ด้วย SVM
        probs = svm_clf.predict_proba(embedding)[0]
        pred = np.argmax(probs)
        name = le.inverse_transform([pred])[0]
        conf = probs[pred] * 100

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(annotated_frame, f"{name} ({conf:.2f}%)",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    # ==============================
    # ตรวจ "แว่น" ด้วย YOLO
    # ==============================
    results_glasses = model_glasses.predict(frame, conf=0.3, verbose=False)
    for box in results_glasses[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        conf = (float(box.conf[0]))*100
        cls_name = results_glasses[0].names[cls_id]

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(annotated_frame, f"Glasses/{cls_name} {conf:.2f}",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2)

    # ==============================
    # แสดงผล
    # ==============================
    cv2.imshow("Face Recognition + Glasses Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
