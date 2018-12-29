#!/usr/bin/python3

from scaleogr import ScaleOGR
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

def run(ogr):
    print("Testing")
    cap = cv2.VideoCapture('../data/test.mp4')
    if cap.isOpened():
        ret, img = cap.read()
        while ret:
            ret, img = cap.read()
            ogr.process(img)
            time.sleep(0.1)
    else:
        print("Test file not found")

s = ScaleOGR(False, False)


camera = PiCamera()
camera.resolution = (1920,1088)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(1920,1088))

for frame in camera.capture_continuous(rawCapture, format="bgr",  use_video_port=True):
    image = rawCapture.array
    num = s.process(image)
    print('Detected number: {:}'.format(num))
    rawCapture.truncate(0)
    #time.sleep(0.1)
