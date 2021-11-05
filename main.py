import cv2
import datetime
import sys
import numpy as np
import pafy


if __name__ == '__main__':
    # url = "https://www.youtube.com/watch?v=IzbCLBXAGFM"
    # video = pafy.new(url)
    # best = video.getbest(preftype="mp4")
    # face_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')   # haarcascade fullbody model
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')   # haarcascade face model
    # cap = cv2.VideoCapture('./videos/people_walking.mp4') # capture a video in local directory
    cap = cv2.VideoCapture(0) # use the current device's default camera (webcam on laptops) as video resource

    while cap.isOpened():
        _, frame = cap.read()
        count = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 1)
        for (x, y, w, h) in faces:
            count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

        cv2.putText(frame, 'Count:' + str(count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break





