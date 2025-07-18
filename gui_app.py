import cv2
import numpy as np
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
from utils import save_face_data, load_face_data, knn_classifier
import os

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

root = Tk()
root.title("🧠 Face Recognition App")
root.geometry("800x600")
root.configure(bg="#f4f4f4")

# UI Elements
video_label = Label(root, bg="black")
video_label.pack(pady=10)

entry_name = Entry(root, font=("Arial", 14))
entry_name.pack(pady=5)
entry_name.insert(0, "Enter your name...")

btn_frame = Frame(root, bg="#f4f4f4")
btn_frame.pack(pady=10)

status_text = StringVar()
status_label = Label(root, textvariable=status_text, font=("Arial", 12), bg="#f4f4f4")
status_label.pack()

# Global vars
faces, labels, label_map = [], [], {}
class_id = 0
is_training = False
is_recognizing = False
collected = 0

# Button functions
def start_training():
    global is_training, collected, class_id
    name = entry_name.get()
    if not name or name == "Enter your name...":
        messagebox.showwarning("Input Error", "Please enter a valid name.")
        return
    if name not in label_map.values():
        label_map[class_id] = name
        assigned_id = class_id
        class_id += 1
    else:
        assigned_id = list(label_map.keys())[list(label_map.values()).index(name)]

    is_training = True
    collected = 0
    status_text.set(f"Training for {name}...")

    def collect(event=None):
        nonlocal assigned_id
        global faces, labels, collected, is_training
        if collected >= 30:
            save_face_data(name, np.array(faces), np.array(labels), label_map)
            is_training = False
            status_text.set("✅ Training complete!")
            return

        ret, frame = cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rect = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces_rect:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (100, 100))
            faces.append(face_resized.flatten())
            labels.append(assigned_id)
            collected += 1
            status_text.set(f"Training... {collected}/30")
            break

    root.after(500, collect)

def recognize():
    global is_recognizing
    is_recognizing = not is_recognizing
    status_text.set("🧠 Recognition mode ON" if is_recognizing else "🛑 OFF")

def update_video():
    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_rect = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces_rect:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (100, 100)).flatten()

        if is_recognizing:
            trainX, trainY, label_map_local = load_face_data()
            if trainX is not None:
                train = np.hstack((trainX, trainY.reshape(-1, 1)))
                pred_id = knn_classifier(train, face_resized)
                name = label_map_local.get(pred_id, "Unknown")
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(Image.fromarray(img))
    video_label.configure(image=img)
    video_label.image = img
    root.after(30, update_video)

# Buttons
Button(btn_frame, text="📸 Train Face", command=start_training, font=("Arial", 12), width=15, bg="#9be7ff").pack(side=LEFT, padx=5)
Button(btn_frame, text="🧠 Recognize", command=recognize, font=("Arial", 12), width=15, bg="#a5d6a7").pack(side=LEFT, padx=5)
Button(btn_frame, text="❌ Quit", command=root.destroy, font=("Arial", 12), width=10, bg="#ef9a9a").pack(side=LEFT, padx=5)

update_video()
root.mainloop()
