import cv2 as cv
import numpy as np
from simple_facerec import SimpleFacerec as sfc
import threading

# Function to process frames and detect faces
def process_frames(cap, sfr):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect Faces
        face_locations, face_names = sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x1, y2, x2 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            cv.putText(frame, name, (x1, y1 - 10), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        cv.imshow("Frame", frame)

        if cv.waitKey(1) == 27:
            break

# Load face encodings
sfr = sfc()
sfr.load_encoding_images("faces/")

# Load Camera
cap = cv.VideoCapture(0)

# Set camera resolution to 720p
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# Create a thread to process frames
thread = threading.Thread(target=process_frames, args=(cap, sfr))
thread.start()

# Wait for the thread to finish
thread.join()

# Release the camera and close OpenCV windows
cap.release()
cv.destroyAllWindows()
