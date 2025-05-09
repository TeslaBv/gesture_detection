import cv2
import mediapipe as mp
import math

# Inicializar Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

def finger_extended(landmarks, tip, pip):
    # True si el TIP está "por encima" (y menor) que el PIP en el eje Y
    return landmarks[tip].y < landmarks[pip].y

def detect_gesture(landmarks):
    # ¿Cada dedo está extendido?
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

    # Distancia pulgar–índice (para OK y corazón coreano)
    x1, y1 = landmarks[mp_hands.HandLandmark.THUMB_TIP].x, landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    x2, y2 = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    dist_thumb_index = math.hypot(x1 - x2, y1 - y2)

    # Detectar gestos en orden de prioridad
    # 1) Corazón Coreano: pulgar e índice muy cerca + medio, anular y meñique doblados
    if dist_thumb_index < 0.04 and not (middle_ext or ring_ext or pinky_ext):
        return "Corazón Coreano"

    # 2) OK: pulgar e índice juntos + otros 3 extendidos
    if dist_thumb_index < 0.05 and middle_ext and ring_ext and pinky_ext:
        return "OK"

    # 3) Like: solo pulgar extendido
    if thumb_ext and not any([index_ext, middle_ext, ring_ext, pinky_ext]):
        return "Like"

    # 4) Rock and Roll (🤘): índice y meñique extendidos, medio y anular doblados
    if index_ext and pinky_ext and not (middle_ext or ring_ext):
        return "Rock and Roll"

    # 5) Amor y Paz (✌️): índice y medio extendidos, otros doblados
    if index_ext and middle_ext and not (ring_ext or pinky_ext or thumb_ext):
        return "Amor y Paz"

    return "Desconocido"

# Captura de video
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detect_gesture(hand_landmarks.landmark)
            cv2.putText(frame, gesture, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gestos", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
