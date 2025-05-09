import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Configuración del dibujo
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# Índices precisos para detección ocular (MediaPipe FACEMESH_EYES)
BLINK_INDICES = {
    'left_eye': [362, 385, 387, 263, 373, 380],  # Ojo izquierdo
    'right_eye': [33, 160, 158, 133, 153, 144]    # Ojo derecho
}

# Parámetros para detección de parpadeo
EAR_CALIBRATION_FRAMES = 30    # 1 segundo a 30 FPS
EAR_SMOOTHING_WINDOW = 5       # Suavizado temporal
BLINK_THRESHOLD_MULTIPLIER = 0.85
MIN_EAR = 0.15
MAX_EAR = 0.25

# Índices para otros gestos
SMILE_INDICES = {
    'left_mouth_corner': 61,    'right_mouth_corner': 291,
    'upper_lip_center': 13,     'lower_lip_center': 14,
    'upper_lip_top': 0,         'lower_lip_bottom': 17
}

HEAD_TURN_INDICES = {
    'nose_tip': 1,     'left_face': 127,    'right_face': 356,
    'face_top': 10,    'face_bottom': 152,  'nose_bridge': 6,
    'forehead': 9,     'chin': 199,         'nose_base': 4  # Nuevo punto añadido
}

OPEN_MOUTH_INDICES = {
    'upper_lip_center': 13,    'lower_lip_center': 14,
    'left_mouth_corner': 78,   'right_mouth_corner': 308,
    'upper_inner_lip': 12,     'lower_inner_lip': 15
}

class EyeBlinkDetector:
    def __init__(self):
        self.ear_history = deque(maxlen=EAR_SMOOTHING_WINDOW)
        self.calibration_data = []
        self.ear_threshold = 0.2
        self.blink_counter = 0
        self.last_blink_time = 0
        self.eyes_closed_start = 0
        self.calibrated = False

    def calculate_ear(self, eye_points):
        """Calcula el Eye Aspect Ratio (EAR) usando 6 puntos."""
        if len(eye_points) < 6: return 0.0
        
        # Convertir landmarks a coordenadas 2D
        points = [(point.x, point.y) for point in eye_points]
        
        # Cálculo de distancias
        v1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
        v2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
        h = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
        
        return (v1 + v2) / (2.0 * h) if h != 0 else 0.0

    def detect_blink(self, landmarks):
        """Detección mejorada con calibración automática."""
        if not landmarks: return False, 0.0
        
        # Obtener puntos oculares
        left_eye = [landmarks[i] for i in BLINK_INDICES['left_eye']]
        right_eye = [landmarks[i] for i in BLINK_INDICES['right_eye']]
         
        # Calcular EAR
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Calibración automática
        if not self.calibrated:
            self.calibration_data.append(avg_ear)
            if len(self.calibration_data) >= EAR_CALIBRATION_FRAMES:
                base_ear = np.mean(self.calibration_data)
                self.ear_threshold = max(MIN_EAR, min(base_ear * BLINK_THRESHOLD_MULTIPLIER, MAX_EAR))
                self.calibrated = True
            return False, 0.0
        
        # Suavizado temporal
        self.ear_history.append(avg_ear)
        smoothed_ear = np.mean(self.ear_history)
        
        # Detección y confianza
        is_blinking = smoothed_ear < self.ear_threshold
        confidence = max(0.0, (self.ear_threshold - smoothed_ear) / (self.ear_threshold - MIN_EAR))
        
        # Detección de ojos cerrados prolongados
        if is_blinking:
            if self.eyes_closed_start == 0:
                self.eyes_closed_start = time.time()
            elif time.time() - self.eyes_closed_start > 0.8:
                confidence = 1.0
        else:
            self.eyes_closed_start = 0
        
        # Contar parpadeos válidos
        if confidence > 0.15 and (time.time() - self.last_blink_time) > 0.2:
            self.blink_counter += 1
            self.last_blink_time = time.time()
        
        return is_blinking, confidence

# Funciones para otros gestos
def detect_smile(landmarks):
    """Detecta una sonrisa basada en múltiples características faciales."""
    # Extraer puntos relevantes
    left_corner = landmarks[SMILE_INDICES['left_mouth_corner']]
    right_corner = landmarks[SMILE_INDICES['right_mouth_corner']]
    upper_lip_center = landmarks[SMILE_INDICES['upper_lip_center']]
    lower_lip_center = landmarks[SMILE_INDICES['lower_lip_center']]
    upper_lip_top = landmarks[SMILE_INDICES['upper_lip_top']]
    lower_lip_bottom = landmarks[SMILE_INDICES['lower_lip_bottom']]
    
    # Altura de la boca (distancia vertical entre labios)
    mouth_height = abs(upper_lip_center.y - lower_lip_center.y)
    full_mouth_height = abs(upper_lip_top.y - lower_lip_bottom.y)
    
    # Ancho de la boca (distancia entre esquinas)
    mouth_width = abs(right_corner.x - left_corner.x)
    
    # Curvatura de las comisuras (positiva si las esquinas están más altas que el centro)
    corner_lift_left = upper_lip_center.y - left_corner.y
    corner_lift_right = upper_lip_center.y - right_corner.y
    corner_lift = (corner_lift_left + corner_lift_right) / 2
    
    # Forma de la boca - estiramiento horizontal
    stretch_factor = mouth_width / full_mouth_height if full_mouth_height > 0 else 0
    
    # Sonrisa con dientes: la boca tiende a abrirse más
    teeth_visible = mouth_height > 0.015 and stretch_factor > 2.0
    
    # Combinar factores para determinar si hay sonrisa
    smile_score = 0
    
    # Para sonrisa normal (sin mostrar dientes)
    if not teeth_visible:
        smile_score = corner_lift * 5  # Más peso a la elevación de las comisuras
    else:
        # Para sonrisa con dientes
        smile_score = (corner_lift * 3) + (stretch_factor / 10)
    
    threshold = 0.02  # Umbral ajustado
    is_smiling = smile_score > threshold or teeth_visible
    confidence = min(smile_score / threshold, 1.0) if is_smiling else 0.0
    
    # Incrementar confianza si los dientes son visibles
    if teeth_visible:
        confidence = max(confidence, 0.7)
    
    return is_smiling, confidence

def detect_head_turn(landmarks):
    """Detecta movimientos de la cabeza con calibración mejorada."""
    # Extraer puntos clave
    nose_tip = landmarks[HEAD_TURN_INDICES['nose_tip']]
    left_face = landmarks[HEAD_TURN_INDICES['left_face']]
    right_face = landmarks[HEAD_TURN_INDICES['right_face']]
    face_top = landmarks[HEAD_TURN_INDICES['face_top']]
    face_bottom = landmarks[HEAD_TURN_INDICES['face_bottom']]
    forehead = landmarks[HEAD_TURN_INDICES['forehead']]
    chin = landmarks[HEAD_TURN_INDICES['chin']]
    
    # Cálculo para movimiento horizontal (izquierda/derecha)
    face_width = abs(right_face.x - left_face.x)
    ideal_center_x = (left_face.x + right_face.x) / 2
    nose_offset_x = (nose_tip.x - ideal_center_x) / (face_width / 2) if face_width > 0 else 0
    
    # Cálculo mejorado para movimiento vertical (arriba/abajo)
    # Usar múltiples puntos de referencia para mayor estabilidad
    face_height = abs(face_top.y - face_bottom.y)
    
    # Posición de la nariz relativa a puntos faciales clave
    neutral_position = 0.45
    vertical_position = (nose_tip.y - face_top.y) / face_height if face_height > 0 else neutral_position
    vertical_offset = vertical_position - neutral_position
    
    # Usar más puntos faciales para mejor detección vertical
    # Ángulo entre frente, nariz y barbilla
    forehead_to_nose_y = nose_tip.y - forehead.y
    nose_to_chin_y = chin.y - nose_tip.y
    
    # Relación entre estos segmentos (menor cuando la cabeza está hacia arriba)
    vertical_angle_factor = forehead_to_nose_y / nose_to_chin_y if nose_to_chin_y > 0 else 1.0
    
    # Umbrales para cada dirección
    threshold_horizontal = 0.2
    threshold_vertical = 0.1  # Más sensible para movimientos verticales
    
    # Determinar dirección principal y confianza
    abs_h_offset = abs(nose_offset_x)
    abs_v_offset = abs(vertical_offset)
    
    # Priorizar movimiento vertical cuando ángulo es significativo
    if vertical_angle_factor < 0.7 and forehead_to_nose_y < nose_to_chin_y * 0.7:
        # Cabeza claramente hacia arriba
        return True, min(1.0, (0.7 - vertical_angle_factor) * 5), "arriba"
    
    elif vertical_angle_factor > 1.3 and forehead_to_nose_y > nose_to_chin_y * 1.3:
        # Cabeza claramente hacia abajo
        return True, min(1.0, (vertical_angle_factor - 1.3) * 3), "abajo"
    
    elif abs_h_offset > threshold_horizontal:
        # Movimiento horizontal
        if nose_offset_x < -threshold_horizontal:
            return True, min(abs_h_offset / threshold_horizontal, 1.0), "izquierda"
        elif nose_offset_x > threshold_horizontal:
            return True, min(abs_h_offset / threshold_horizontal, 1.0), "derecha"
    
    # Si no hay giro significativo
    return False, 0.0, "centro"

def detect_open_mouth(landmarks):
    """Detecta apertura de boca con puntos internos."""
    upper = landmarks[OPEN_MOUTH_INDICES['upper_lip_center']]
    lower = landmarks[OPEN_MOUTH_INDICES['lower_lip_center']]
    inner_upper = landmarks[OPEN_MOUTH_INDICES['upper_inner_lip']]
    inner_lower = landmarks[OPEN_MOUTH_INDICES['lower_inner_lip']]
    
    # Cálculo de apertura vertical
    outer_height = abs(upper.y - lower.y)
    inner_height = abs(inner_upper.y - inner_lower.y)
    openness = (outer_height + inner_height) / 2
    
    threshold = 0.035
    is_open = openness > threshold
    confidence = min(openness / threshold, 1.0)
    
    return is_open, confidence

def main():
    cap = cv2.VideoCapture(0)
    blink_detector = EyeBlinkDetector()
    
    gestures = ["smile", "blink", "head_turn", "open_mouth"]
    gesture_names = ["😄 Sonrisa", "👁 Parpadeo", "↔ Giro Cabeza", "👄 Boca Abierta"]
    current_gesture = 0
    
    # Historial para suavizado
    history = {gesture: [] for gesture in gestures}
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        detection_status = "No detectado"
        confidence = 0.0
        color = (200, 200, 200)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = face_landmarks.landmark
            
            # Detectar gesto actual
            if gestures[current_gesture] == "smile":
                is_detected, confidence = detect_smile(landmarks)
                detection_status = "Sonriendo" if is_detected else "Cara neutral"
                color = (0, 255, 0) if is_detected else (0, 0, 255)
            
            elif gestures[current_gesture] == "blink":
                is_detected, confidence = blink_detector.detect_blink(landmarks)
                eye_state = "Ojos Cerrados" if confidence > 0.9 else "Parpadeando"
                detection_status = f"{eye_state} ({blink_detector.blink_counter})"
                color = (0, 0, 255) if confidence > 0.9 else (0, 165, 255)
            
            elif gestures[current_gesture] == "head_turn":
                is_detected, conf, direction = detect_head_turn(landmarks)
                detection_status = f"Mirando {direction}" if is_detected else "Centrado"
                confidence = conf
                color = (255, 0, 0) if is_detected else (200, 200, 200)
            
            elif gestures[current_gesture] == "open_mouth":
                is_detected, confidence = detect_open_mouth(landmarks)
                detection_status = "Boca Abierta" if is_detected else "Boca Cerrada"
                color = (0, 255, 255) if is_detected else (0, 0, 255)
            
            # Suavizar detección
            history[gestures[current_gesture]].append((is_detected, confidence))
            if len(history[gestures[current_gesture]]) > 5:
                history[gestures[current_gesture]].pop(0)
            
            # Dibujar landmarks
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec
            )
        
        # Interfaz de usuario
        cv2.putText(frame, f"Detectando: {gesture_names[current_gesture]}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Estado: {detection_status}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Confianza: {confidence:.2f}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imshow('Detección de Gestos Avanzada', frame)
        
        # Controles
        key = cv2.waitKey(5)
        if key == 27:  # ESC
            break
        elif key == 32:  # Espacio
            current_gesture = (current_gesture + 1) % len(gestures)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()