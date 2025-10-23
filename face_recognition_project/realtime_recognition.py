# realtime_recognition.py
import cv2
import numpy as np
from face_recognition import load_encodings
import face_recognition

def recognize_webcam(tolerance=0.6):
    cap = cv2.VideoCapture(0)

    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam. Please check if it's connected or in use by another program.")
        return

    encodings_dict = load_encodings()
    if not encodings_dict:
        print("Warning: No encodings found. Make sure you've registered users.")
    
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to grab frame. Retrying...")
            continue  # try next frame

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            user_ids = list(encodings_dict.keys())
            stored_encodings = np.array(list(encodings_dict.values()))
            
            # Handle empty encodings_dict
            if len(stored_encodings) == 0:
                name = "Unknown"
            else:
                distances = np.linalg.norm(stored_encodings - encoding, axis=1)
                idx = np.where(distances <= tolerance)[0]
                name = user_ids[idx[0]] if len(idx) > 0 else 'Unknown'

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow('Real-time Recognition', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_webcam()
