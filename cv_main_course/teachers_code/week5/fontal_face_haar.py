import numpy as np
import cv2
import sys
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture('ming.mp4')
# cap = cv2.VideoCapture(sys.argv[1])
while cap.isOpened():

    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # roi_gray = gray[y:y + h, x:x + w]
            # roi_color = frame[y:y + h, x:x + w]
            # eyes = eye_cascade.detectMultiScale(roi_gray)
            # for (ex, ey, ew, eh) in eyes:
            #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        frame = cv2.resize(frame, (400, 200), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(2)
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
