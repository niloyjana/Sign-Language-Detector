import cv2
import pickle
import numpy as np
import mediapipe as mp

# =============================
# CONFIG
# =============================
MODEL_PATH = "model.p"   # your trained model
CONF_THRESHOLD = 0.0     # keep low, UI shows confidence

BOX_SCALE = 0.6          # 40% smaller UI
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
FONT_THICKNESS = 1

# =============================
# LOAD MODEL
# =============================
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# =============================
# MEDIAPIPE SETUP
# =============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# =============================
# CAMERA
# =============================
cap = cv2.VideoCapture(0)

# =============================
# HELPERS
# =============================
def get_hand_label(handedness):
    label = handedness.classification[0].label
    # camera is NOT flipped → mediapipe is correct
    return "Right" if label == "Right" else "Left"

def draw_hand_box(img, landmarks):
    h, w, _ = img.shape
    xs = [int(lm.x * w) for lm in landmarks.landmark]
    ys = [int(lm.y * h) for lm in landmarks.landmark]

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    pad = int(20 * BOX_SCALE)
    x1 -= pad
    y1 -= pad
    x2 += pad
    y2 += pad

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
    return x1, y1, x2, y2

def extract_keypoints(hand_landmarks):
    data = []
    for lm in hand_landmarks.landmark:
        data.extend([lm.x, lm.y])
    return np.array(data).reshape(1, -1)

# =============================
# MAIN LOOP
# =============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror for user
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_lms, handedness in zip(
            result.multi_hand_landmarks,
            result.multi_handedness
        ):
            # ---- DRAW FINGER TRACKING (IMPORTANT)
            mp_draw.draw_landmarks(
                frame,
                hand_lms,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=2),
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1),
            )

            # ---- HAND BOX
            x1, y1, x2, y2 = draw_hand_box(frame, hand_lms)

            # ---- HAND SIDE
            hand_side = get_hand_label(handedness)

            # ---- MODEL PREDICTION
            keypoints = extract_keypoints(hand_lms)
            probs = model.predict_proba(keypoints)[0]
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            gesture = model.classes_[pred_idx]

            # ---- INFO BOX (HOVER)
            info_y = y1 - int(10 * BOX_SCALE)
            if info_y < 20:
                info_y = y2 + int(20 * BOX_SCALE)

            text = f"{hand_side} | {gesture} ({int(confidence*100)}%)"

            (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
            cv2.rectangle(
                frame,
                (x1, info_y - th - 6),
                (x1 + tw + 6, info_y + 2),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                frame,
                text,
                (x1 + 3, info_y),
                FONT,
                FONT_SCALE,
                (255, 255, 255),
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

    cv2.imshow("Sign Language Detector", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
