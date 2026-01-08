import cv2
import numpy as np
import pygame
import time


pygame.mixer.init()
ALARM_PATH = "alarm.mp3"

Alarm_Status = False
Fire_Reported = 0
last_fire_time = 0
ALARM_COOLDOWN = 1

def play_alarm():
    global Alarm_Status
    if not Alarm_Status:
        pygame.mixer.music.load(ALARM_PATH)
        pygame.mixer.music.play(-1)
        Alarm_Status = True

def stop_alarm():
    global Alarm_Status
    if Alarm_Status:
        pygame.mixer.music.stop()
        Alarm_Status = False


video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_mask = None
REQUIRED_FRAMES = 3

while True:
    grabbed, frame = video.read()
    if not grabbed:
        break

    frame_blur = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)


    lower_fire = np.array([15, 50, 150])
    upper_fire = np.array([35, 255, 255])
    color_mask = cv2.inRange(hsv, lower_fire, upper_fire)

    _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_or(color_mask, bright_mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


    fire_detected = False
    if prev_mask is not None:
        diff_mask = cv2.absdiff(mask, prev_mask)
        if np.count_nonzero(diff_mask) > 10000:
            fire_detected = True
    prev_mask = mask.copy()


    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:
            fire_detected = True
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            break


    if fire_detected:
        Fire_Reported += 1
    else:
        Fire_Reported = 0
        stop_alarm()

    if Fire_Reported >= REQUIRED_FRAMES:
        current_time = time.time()
        if current_time - last_fire_time > ALARM_COOLDOWN:
            play_alarm()
            last_fire_time = current_time
        cv2.putText(frame, "🔥 FIRE DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)


    cv2.imshow("Fire Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


stop_alarm()
video.release()
cv2.destroyAllWindows()
