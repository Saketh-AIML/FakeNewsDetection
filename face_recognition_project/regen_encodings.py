# regen_encodings.py
from face_recognition import add_user
import os

faces_path = 'faces'
for user_id in os.listdir(faces_path):
    user_folder = os.path.join(faces_path, user_id)
    if os.path.isdir(user_folder):
        for img_file in os.listdir(user_folder):
            img_path = os.path.join(user_folder, img_file)
            add_user(user_id, img_path)