import cv2
import numpy as np
import pygame
import time

# ---------------- AUDIO SETUP ----------------
pygame.mixer.init()
ALARM_PATH = "alarm.mp3"

alarm_on = False
last_fire_time = 0
ALARM_COOLDOWN = 1  # seconds between alarms

def start_alarm():
    global alarm_on
    if not alarm_on:
        pygame.mixer.music.load(ALARM_PATH)
        pygame.mixer.music.play(-1)
        alarm_on = True

def stop_alarm():
    global alarm_on
    if alarm_on:
        pygame.mixer.music.stop()
        alarm_on = False

# ---------------- CAMERA SETUP ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fire_frames = 0
REQUIRED_FRAMES = 3  # small lighter flames appear fast

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Fast blur
    frame_blur = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)

    # Fire color mask (yellow-orange)
    lower_fire = np.array([15, 50, 150])
    upper_fire = np.array([35, 255, 255])
    color_mask = cv2.inRange(hsv, lower_fire, upper_fire)

    # Brightness mask (for tiny flames)
    _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Combine masks
    mask = cv2.bitwise_or(color_mask, bright_mask)

    # Remove noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Detect contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fire_detected = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:  # lowered threshold for tiny flames
            fire_detected = True
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            break

    # ---------------- FIRE LOGIC ----------------
    if fire_detected:
        fire_frames += 1
    else:
        fire_frames = 0
        stop_alarm()

    if fire_frames >= REQUIRED_FRAMES:
        current_time = time.time()
        if current_time - last_fire_time > ALARM_COOLDOWN:
            start_alarm()
            last_fire_time = current_time
        cv2.putText(frame, "🔥 FIRE DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    # ---------------- DISPLAY ----------------
    cv2.imshow("Fire Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
stop_alarm()
cap.release()
cv2.destroyAllWindows()
