import cv2
import mediapipe as mp
import numpy as np
import time
import random
from collections import deque

# ================= CONFIG =================
WIDTH, HEIGHT = 1280, 720
SMOOTHING = 5
CLEAR_COOLDOWN = 2

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# ================= CANVAS =================
canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

# ================= COLORS =================
COLORS = [(255,0,255),(0,255,255),(255,255,0),(0,255,0),(255,0,0)]
color_i = 0
current_color = COLORS[color_i]

brushes = ["Pen", "Neon", "Glow"]
brush_i = 0
current_brush = brushes[brush_i]

brush_size = 5
eraser_size = 40

# ================= STATE =================
points = deque(maxlen=SMOOTHING)
prev_x, prev_y = 0, 0
last_clear = 0


# ================= FUNCTIONS =================

def fingers(lm):
    f = []
    f.append(1 if lm[4].x < lm[3].x else 0)
    tips = [8, 12, 16, 20]
    for t in tips:
        f.append(1 if lm[t].y < lm[t-2].y else 0)
    return f


def smooth(x, y):
    points.append((x, y))
    return int(np.mean([p[0] for p in points])), int(np.mean([p[1] for p in points]))


def draw(canvas, x1, y1, x2, y2):
    if current_brush == "Pen":
        cv2.line(canvas,(x1,y1),(x2,y2),current_color,brush_size)

    elif current_brush == "Neon":
        cv2.line(canvas,(x1,y1),(x2,y2),current_color,12)
        cv2.line(canvas,(x1,y1),(x2,y2),(255,255,255),2)

    elif current_brush == "Glow":
        overlay = canvas.copy()
        cv2.line(overlay,(x1,y1),(x2,y2),current_color,20)
        canvas[:] = cv2.addWeighted(overlay,0.3,canvas,0.7,0)


def ui(frame, mode):
    cv2.rectangle(frame,(0,0),(WIDTH,80),(30,30,30),-1)
    cv2.circle(frame,(60,40),15,current_color,-1)

    cv2.putText(frame,f"Brush:{current_brush}",(120,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    cv2.putText(frame,f"Mode:{mode}",(120,65),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)


# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    mode = "Idle"

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        f = fingers(lm)

        x = int(lm[8].x * WIDTH)
        y = int(lm[8].y * HEIGHT)
        x, y = smooth(x, y)

        # ================= DRAW =================
        if f == [0,1,0,0,0]:
            mode = "DRAW"
            if prev_x == 0:
                prev_x, prev_y = x, y

            draw(canvas, prev_x, prev_y, x, y)
            prev_x, prev_y = x, y

        # ================= NAV =================
        elif f == [0,1,1,0,0]:
            mode = "NAV"
            prev_x, prev_y = 0,0

        # ================= ERASE =================
        elif f == [0,0,0,0,0]:
            mode = "ERASE"
            cv2.circle(canvas,(x,y),eraser_size,(0,0,0),-1)

        # ================= CLEAR =================
        elif f == [1,1,1,1,1]:
            mode = "CLEAR"
            if time.time() - last_clear > CLEAR_COOLDOWN:
                canvas = np.zeros((HEIGHT,WIDTH,3),dtype=np.uint8)
                last_clear = time.time()

    # ================= MERGE =================
    gray = cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY)
    _,mask = cv2.threshold(gray,20,255,cv2.THRESH_BINARY)

    inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(frame,frame,mask=inv)
    fg = cv2.bitwise_and(canvas,canvas,mask=mask)

    final = cv2.add(bg,fg)

    ui(final, mode)

    cv2.imshow("Gesture Board", final)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('s'):
        cv2.imwrite(f"draw_{int(time.time())}.png", canvas)
        print("Saved!")

    elif key == ord('c'):
        color_i = (color_i+1)%len(COLORS)
        current_color = COLORS[color_i]

    elif key == ord('b'):
        brush_i = (brush_i+1)%len(brushes)
        current_brush = brushes[brush_i]

cap.release()
cv2.destroyAllWindows()
