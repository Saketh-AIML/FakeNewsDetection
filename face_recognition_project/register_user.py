# register_user.py
import os
import cv2
import face_recognition
from face_utils import encode_face_multi, load_encodings, save_encodings

DATA_FOLDER = "data/users"

def capture_and_recognize(username, num_images=5):
    """
    Capture face images from webcam for a new user and update encodings automatically.
    """
    user_folder = os.path.join(DATA_FOLDER, username)
    os.makedirs(user_folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print(f"[INFO] Capturing {num_images} images for {username}. Press 's' to save each image.")

    captured_count = 0
    encodings_dict = load_encodings()

    while captured_count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame. Retrying...")
            continue

        cv2.imshow(f"Capturing for {username}", frame)
        key = cv2.waitKey(1) & 0xFF

        # Press 's' to save the image
        if key == ord('s'):
            img_path = os.path.join(user_folder, f"{username}_{captured_count+1}.jpg")
            cv2.imwrite(img_path, frame)
            
            # Auto encode the captured image
            encoding = encode_face_multi(img_path)
            if encoding is not None:
                encodings_dict[username] = encoding
                save_encodings(encodings_dict)
                print(f"[INFO] Saved and encoded image {captured_count+1}/{num_images}")
                captured_count += 1
            else:
                print("[WARNING] Could not detect a face. Try again.")

        # Press 'q' to quit early
        elif key == ord('q'):
            print("[INFO] Exiting capture early.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Registration complete for {username}")
