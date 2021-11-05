import torch
import numpy as np
import cv2
import pafy
from time import time



if __name__ == '__main__':
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s6', pretrained=True)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./weights/crowdhuman_yolov5m.pt')
    # model = torch.hub.load('ultralytics/yolov5', 'custom', path='./weights/body1500_yolov5s6.pt')
    model.classes = [0]
    classes = model.names
    # cap = cv2.VideoCapture(0) # use the current device's default camera (webcam on laptops) as video resource
    cap = cv2.VideoCapture('./videos/people_walking.mp4') # capture a video in local directory
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    while cap.isOpened():
        _, frame = cap.read()
        count = 0

        # Score frames
        frames = [frame]
        results = model(frames)
        labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        unique, counts = np.unique(labels, return_counts=True)
        print(dict(zip(unique, counts)))

        # Plot boxes
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1)
                cv2.putText(frame, classes[int(labels[i])], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1)

        cv2.imshow('Webcam', frame)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break