# face_utils.py
import os
import pickle
import cv2
import face_recognition
import numpy as np

DATA_PATH = 'data/encodings.pkl'

def load_encodings():
    if not os.path.exists(DATA_PATH):
        return {}
    if os.path.getsize(DATA_PATH) == 0:
        return {}
    try:
        with open(DATA_PATH, 'rb') as f:
            return pickle.load(f)
    except EOFError:
        print("[WARNING] Encodings file is empty or corrupted. Starting fresh.")
        return {}


def save_encodings(encodings):
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, 'wb') as f:
        pickle.dump(encodings, f)

def encode_face_multi(image_path):
    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    if encodings:
        return encodings[0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_img = rgb[y:y+h, x:x+w]
        encodings = face_recognition.face_encodings(face_img)
        if encodings:
            return encodings[0]
    return None