import cv2
import os
import sys

user_name = input("Enter user name: ")
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

save_path = f"dataset/{user_name}"
os.makedirs(save_path, exist_ok=True)

count = 0
while count < 100:
    ret, frame = cap.read()
    if not ret:
        break
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in faces:
        count += 1
        face = frame[y:y+h, x:x+w]
        cv2.imwrite(f"{save_path}/{count}.jpg", face)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow("Collecting Faces", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
