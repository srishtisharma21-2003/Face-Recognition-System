import os
import numpy as np
import cv2
import pickle

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_face_data(name, faces, labels, label_map):
    data_path = "dataset"
    ensure_folder(data_path)
    np.save(os.path.join(data_path, "face_data.npy"), faces)
    np.save(os.path.join(data_path, "face_labels.npy"), labels)
    with open(os.path.join(data_path, "label_map.npy"), "wb") as f:
        pickle.dump(label_map, f)

def load_face_data():
    try:
        faces = np.load("dataset/face_data.npy")
        labels = np.load("dataset/face_labels.npy")
        with open("dataset/label_map.npy", "rb") as f:
            label_map = pickle.load(f)
        return faces, labels, label_map
    except:
        return None, None, {}

def knn_classifier(train, test, k=5):
    distances = []
    for i in range(len(train)):
        dist = np.linalg.norm(train[i][:-1] - test)
        distances.append((dist, train[i][-1]))
    distances = sorted(distances)[:k]
    labels = [d[1] for d in distances]
    output = max(set(labels), key=labels.count)
    return output
