import mediapipe
import cv2
import keyboard
import pyautogui
import autoit
import pydirectinput

capture = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

mpHands = mediapipe.solutions.hands
mpDraw = mediapipe.solutions.drawing_utils
hands = mpHands.Hands(model_complexity=0)

RESOLUTION = 854 * 480
MONITOR_RES = 3840 * 2160

MULTIPLIER = 1

pyautogui.FAILSAFE = False

while True:
    ret, frame = capture.read()

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    landmarks = results.multi_hand_landmarks



    if landmarks and landmarks[0]:
        for handLms in landmarks:
            mpDraw.draw_landmarks(frame, handLms)

        lm = landmarks[0].landmark[0]
        x, y = lm.x * 3840, lm.y * 2160
        cv2.putText(frame, f"x: {lm.x}, y: {lm.y}",(10, 70) ,cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0))
        pydirectinput.moveTo(int(x), int(y))

    cv2.imshow("frame", frame)

    cv2.resizeWindow("frame", 854, 480)

    if cv2.waitKey(1) == ord("q") or keyboard.is_pressed("q"):
        break


capture.release()
