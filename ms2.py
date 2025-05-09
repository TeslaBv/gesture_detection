from fastapi import FastAPI, UploadFile, File
from typing import List, Dict
import uvicorn
import cv2
import numpy as np
import mediapipe as mp
import math
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Hand Gesture Detection API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Inicializar Mediapipe Hands globalmente
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=1,
                       min_detection_confidence=0.5)

# Lista de posibles gestos
GESTURES = ["Corazón Coreano", "OK", "Like", "Rock and Roll", "Amor y Paz"]

# Funciones auxiliares
def finger_extended(landmarks, tip, pip):
    return landmarks[tip].y < landmarks[pip].y

def detect_gesture(landmarks) -> str:
    thumb_ext = landmarks[mp_hands.HandLandmark.THUMB_TIP].x < landmarks[mp_hands.HandLandmark.THUMB_IP].x
    index_ext = finger_extended(landmarks,
                                mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                mp_hands.HandLandmark.INDEX_FINGER_PIP)
    middle_ext = finger_extended(landmarks,
                                 mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                 mp_hands.HandLandmark.MIDDLE_FINGER_PIP)
    ring_ext = finger_extended(landmarks,
                               mp_hands.HandLandmark.RING_FINGER_TIP,
                               mp_hands.HandLandmark.RING_FINGER_PIP)
    pinky_ext = finger_extended(landmarks,
                                mp_hands.HandLandmark.PINKY_TIP,
                                mp_hands.HandLandmark.PINKY_PIP)

    x1, y1 = landmarks[mp_hands.HandLandmark.THUMB_TIP].x, landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    x2, y2 = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    dist_thumb_index = math.hypot(x1 - x2, y1 - y2)

    if dist_thumb_index < 0.04 and not (middle_ext or ring_ext or pinky_ext):
        return "Corazón Coreano"
    if dist_thumb_index < 0.05 and middle_ext and ring_ext and pinky_ext:
        return "OK"
    if thumb_ext and not any([index_ext, middle_ext, ring_ext, pinky_ext]):
        return "Like"
    if index_ext and pinky_ext and not (middle_ext or ring_ext):
        return "Rock and Roll"
    if index_ext and middle_ext and not (ring_ext or pinky_ext or thumb_ext):
        return "Amor y Paz"
    return "Desconocido"

@app.post("/detect_batch", response_model=Dict[str, int])
def detect_batch(images: List[UploadFile] = File(...)):
    gesture_flags = {gesture: 0 for gesture in GESTURES}

    for img_file in images:
        try:
            data = img_file.file.read()
            np_arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resp = hands.process(rgb)

            if not resp.multi_hand_landmarks:
                continue

            landmarks = resp.multi_hand_landmarks[0].landmark
            gesture = detect_gesture(landmarks)
            if gesture in gesture_flags:
                gesture_flags[gesture] = 1  # Activamos la bandera
        except Exception as e:
            print(f"Error con {img_file.filename}: {e}")
        finally:
            img_file.file.close()

    return gesture_flags

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
