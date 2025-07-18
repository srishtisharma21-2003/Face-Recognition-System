import os
import cv2
import numpy as np
import pickle

from utils import encode_face

data, labels, label_map = [], [], {}
label_id = 0

dataset_dir = "dataset"
for person in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person)
    if not os.path.isdir(person_path): continue

    for image_name in os.listdir(person_path):
        img_path = os.path.join(person_path, image_name)
        encoding = encode_face(img_path)
        if encoding is not None:
            data.append(encoding)
            labels.append(label_id)
    label_map[label_id] = person
    label_id += 1

np.save("face_data.npy", data)
np.save("face_labels.npy", labels)
np.save("label_map.npy", label_map)

print("Training completed and saved.")
