import pandas as pd
import numpy as np 
import cv2
from mtcnn import MTCNN
import os
import csv
# Load data
data_path = "./data"
output_path = "faces_data.csv"
def read_image(file_path):
    """
    Doc anh trong data
    """
    images = []
    labels = []
    for person_name in sorted(os.listdir(file_path)):
        person_folder = os.path.join(file_path, person_name)
        if os.path.isdir(person_folder):
            for image_name in sorted(os.listdir(person_folder)):
                image_path = os.path.join(person_folder, image_name)
                if image_name.endswith('.jpg') or image_name.endswith('.png'):
                    img = cv2.imread(image_path)
                    if img is not None:
                        images.append(img)
                        labels.append(person_name)
    return images, labels

def face_detection(images, labels):
    """
    Phat hien mat trong anh
    """
    detector = MTCNN()
    faces_data = []
    labels_data = []
    for image, label in zip(images, labels):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(image_rgb)
        for face in faces:
            x,y, width, height = face['box']
            face_crop = image[y:y+height, x: x + width]
            face_crop = cv2.resize(face_crop, (160, 160))
            faces_data.append(face_crop)
            labels_data.append(label)
    return faces_data, labels_data

def save_data(faces_data, labels_data, output_path):
    """
    Luu du lieu da xu ly vao file csv 
    """
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'image'])
        for label, face in zip(labels_data, faces_data):
            face_flat = face.flatten()
            writer.writerow([label, np.array(face_flat)])
    print(f"Data saved to {output_path}")
