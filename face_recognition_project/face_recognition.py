# face_recognition.py
from face_utils import load_encodings, save_encodings, encode_face_multi
import numpy as np

def add_user(user_id, image_path):
    encodings = load_encodings()
    encoding = encode_face_multi(image_path)
    if encoding is None:
        raise ValueError("No face found in the image")
    encodings[user_id] = encoding
    save_encodings(encodings)
    print(f"[INFO] User {user_id} added.")

def remove_user(user_id):
    encodings = load_encodings()
    if user_id in encodings:
        del encodings[user_id]
        save_encodings(encodings)
        print(f"[INFO] User {user_id} removed.")

def update_user(user_id, image_path):
    encodings = load_encodings()
    if user_id in encodings:
        encoding = encode_face_multi(image_path)
        encodings[user_id] = encoding
        save_encodings(encodings)
        print(f"[INFO] User {user_id} updated.")
    else:
        print("[WARN] User not found. Use add_user to register.")

def list_users():
    encodings = load_encodings()
    return list(encodings.keys())

def recognize_face(image_path, tolerance=0.6):
    unknown_encoding = encode_face_multi(image_path)
    if unknown_encoding is None:
        return None

    encodings_dict = load_encodings()
    if not encodings_dict:
        return "Unknown"

    user_ids = list(encodings_dict.keys())
    encodings = np.array(list(encodings_dict.values()))
    distances = np.linalg.norm(encodings - unknown_encoding, axis=1)
    idx = np.where(distances <= tolerance)[0]

    if len(idx) > 0:
        return user_ids[idx[0]]
    return "Unknown"