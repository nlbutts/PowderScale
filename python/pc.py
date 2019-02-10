#!/usr/bin/python3

from scaleogr import ScaleOGR
import cv2
import time

def run(ogr):
    print("Testing")
    cap = cv2.VideoCapture(r'C:\ztemp\test.mp4')
    #cap = cv2.VideoCapture('../data/test.mp4')
    if cap.isOpened():
        ret, img = cap.read()
        while ret:
            ret, img = cap.read()
            ogr.process(img)
            time.sleep(0.1)
    else:
        print("Test file not found")

s = ScaleOGR(True)

#s.run_training()
run(s)
