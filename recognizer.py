import cv2
import numpy as np
from utils import get_knn_model, detect_face

knn = get_knn_model()
label_map = np.load("label_map.npy", allow_pickle=True).item()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    faces = detect_face(frame)
    for (x, y, w, h) in faces:
        face_img = cv2.resize(frame[y:y+h, x:x+w], (100, 100)).flatten().reshape(1, -1)
        pred = knn.predict(face_img)
        name = label_map.get(pred[0], "Unknown")
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow("Recognizer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
