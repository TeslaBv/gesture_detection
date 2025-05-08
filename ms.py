from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import mediapipe as mp
from collections import deque, Counter
import time

# ----- MediaPipe setup -----
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# ----- Landmark indices -----
BLINK_INDICES = {'left_eye': [362, 385, 387, 263, 373, 380], 'right_eye': [33, 160, 158, 133, 153, 144]}
SMILE_INDICES = {'left_mouth_corner': 61, 'right_mouth_corner': 291, 'upper_lip_center': 13, 'lower_lip_center': 14,
                 'upper_lip_top': 0, 'lower_lip_bottom': 17}
HEAD_TURN_INDICES = {'nose_tip': 1, 'left_face': 127, 'right_face': 356, 'face_top': 10, 'face_bottom': 152,
                     'nose_bridge': 6, 'forehead': 9, 'chin': 199, 'nose_base': 4}
OPEN_MOUTH_INDICES = {'upper_lip_center': 13, 'lower_lip_center': 14, 'left_mouth_corner': 78, 'right_mouth_corner': 308,
                      'upper_inner_lip': 12, 'lower_inner_lip': 15}


# ----- Gesture detectors -----
class EyeBlinkDetector:
    def __init__(self):
        self.history = deque(maxlen=5)
        self.cal_data = []
        self.threshold = 0.2
        self.calibrated = False

    def calc_ear(self, pts):
        p = [(pt.x, pt.y) for pt in pts]
        v1 = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
        v2 = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
        h = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
        return 0 if h == 0 else (v1 + v2) / (2 * h)

    def detect(self, lm):
        left = [lm[i] for i in BLINK_INDICES['left_eye']]
        right = [lm[i] for i in BLINK_INDICES['right_eye']]
        ear = (self.calc_ear(left) + self.calc_ear(right)) / 2
        if not self.calibrated:
            self.cal_data.append(ear)
            if len(self.cal_data) >= 30:
                base = np.mean(self.cal_data)
                self.threshold = max(0.15, min(base * 0.85, 0.25))
                self.calibrated = True
            return False, 0.0
        self.history.append(ear)
        smooth = np.mean(self.history)
        blink = smooth < self.threshold
        conf = max(0.0, (self.threshold - smooth) / (self.threshold - 0.15))
        return blink, round(conf, 2)


def detect_smile(lm):
    p = lambda i: lm[i]
    lc, rc = p(SMILE_INDICES['left_mouth_corner']), p(SMILE_INDICES['right_mouth_corner'])
    upc, loc = p(SMILE_INDICES['upper_lip_center']), p(SMILE_INDICES['lower_lip_center'])
    upt, lob = p(SMILE_INDICES['upper_lip_top']), p(SMILE_INDICES['lower_lip_bottom'])
    h = abs(upc.y - loc.y)
    fh = abs(upt.y - lob.y)
    w = abs(rc.x - lc.x)
    lift = ((upc.y - lc.y) + (upc.y - rc.y)) / 2
    stretch = w / fh if fh > 0 else 0
    teeth = h > 0.015 and stretch > 2
    score = (lift * 5) if not teeth else (lift * 3 + stretch / 10)
    thr = 0.02
    smile = score > thr or teeth
    conf = min(score / thr, 1.0) if smile else 0.0
    if teeth: conf = max(conf, 0.7)
    return smile, round(conf, 2)


def detect_head_turn(lm):
    nt = lm[HEAD_TURN_INDICES['nose_tip']]
    lf, rf = lm[HEAD_TURN_INDICES['left_face']], lm[HEAD_TURN_INDICES['right_face']]
    fw = abs(rf.x - lf.x)
    cx = (lf.x + rf.x) / 2
    nx = (nt.x - cx) / (fw / 2) if fw > 0 else 0
    fd = nt.y - lm[HEAD_TURN_INDICES['forehead']].y
    dc = lm[HEAD_TURN_INDICES['chin']].y - nt.y
    vf = fd / dc if dc > 0 else 1
    if vf < 0.7:
        return True, round(min((0.7 - vf) * 5, 1.0), 2), "arriba"
    if vf > 1.3:
        return True, round(min((vf - 1.3) * 3, 1.0), 2), "abajo"
    th = 0.2
    if abs(nx) > th:
        return True, round(min(abs(nx) / th, 1.0), 2), ("izquierda" if nx < 0 else "derecha")
    return False, 0.0, "centro"


def detect_open_mouth(lm):
    u, l = lm[OPEN_MOUTH_INDICES['upper_lip_center']], lm[OPEN_MOUTH_INDICES['lower_lip_center']]
    iu, il = lm[OPEN_MOUTH_INDICES['upper_inner_lip']], lm[OPEN_MOUTH_INDICES['lower_inner_lip']]
    oh, ih = abs(u.y - l.y), abs(iu.y - il.y)
    op = (oh + ih) / 2
    thr = 0.035
    open_m = op > thr
    conf = min(op / thr, 1.0)
    return open_m, round(conf, 2)


# ----- Helper functions -----
def get_landmarks(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    return results.multi_face_landmarks[0].landmark


# ----- FastAPI service -----
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Detector global para mantener estado entre frames
blink_detector = EyeBlinkDetector()

# ---- Endpoints consolidados ----
@app.post("/detect/gestures")
async def detect_gestures(files: list[UploadFile] = File(...)):
    results = {
        "blink": False,
        "smile": False,
        "head_turn": {"detected": False, "direction": None},
        "open_mouth": False,
        "nodding": False
    }

    # Contadores para gestos acumulativos
    blink_frames = 0
    smile_frames = 0
    open_mouth_frames = 0
    head_up = 0
    head_down = 0
    head_left = 0
    head_right = 0

    for file in files:
        data = await file.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        landmarks = get_landmarks(img)
        
        if not landmarks:
            continue

        # Detección de cada gesto
        blink, blink_conf = blink_detector.detect(landmarks)
        if blink and blink_conf > 0.7:
            blink_frames += 1

        smile, smile_conf = detect_smile(landmarks)
        if smile and smile_conf > 0.7:
            smile_frames += 1

        open_mouth, mouth_conf = detect_open_mouth(landmarks)
        if open_mouth and mouth_conf > 0.7:
            open_mouth_frames += 1

        head_turn, _, direction = detect_head_turn(landmarks)
        if head_turn:
            if direction == "arriba":
                head_up += 1
            elif direction == "abajo":
                head_down += 1
            elif direction == "izquierda":
                head_left += 1
            elif direction == "derecha":
                head_right += 1

    # Umbrales para considerar un gesto como detectado
    total_frames = len(files)
    if total_frames == 0:
        return results

    # Blink: Al menos en el 10% de los frames
    results["blink"] = (blink_frames / total_frames) >= 0.1

    # Smile: Al menos en el 30% de los frames
    results["smile"] = (smile_frames / total_frames) >= 0.3

    # Open mouth: Al menos en el 20% de los frames
    results["open_mouth"] = (open_mouth_frames / total_frames) >= 0.2

    # Head turn: Dirección predominante
    head_directions = {
        "arriba": head_up,
        "abajo": head_down,
        "izquierda": head_left,
        "derecha": head_right
    }
    predominant_dir = max(head_directions.items(), key=lambda x: x[1])
    if predominant_dir[1] / total_frames >= 0.15:  # Umbral del 15%
        results["head_turn"] = {
            "detected": True,
            "direction": predominant_dir[0]
        }

    # Nodding: Requiere movimientos arriba+abajo significativos
    results["nodding"] = (head_up / total_frames >= 0.1) and (head_down / total_frames >= 0.1)

    return results

# Función auxiliar para landmarks (igual que antes)
def get_landmarks(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    return results.multi_face_landmarks[0].landmark

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)