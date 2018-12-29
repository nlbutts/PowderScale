#!/usr/bin/python3

from scaleogr import ScaleOGR
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from stepper import Stepper

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

class Filter():
    def __init__(self, gain):
        self.gain = gain
        self.filtered_value = 0

    def filter(self, value):
        err = value - self.filtered_value
        newV = err * self.gain
        self.filtered_value += newV

        return self.filtered_value


set_point = 26.0

ogr = ScaleOGR(False, False)
stepper = Stepper(1000000)
f = Filter(0.5)

camera = PiCamera()
camera.close()
camera = PiCamera()
camera.resolution = (1920,1088)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(1920,1088))

for frame in camera.capture_continuous(rawCapture, format="bgr",  use_video_port=True):
    image = rawCapture.array
    num = ogr.process(image)
    fv = f.filter(num)
    print('Detected number: {:} -- Filtered number: {:}'.format(num, fv))
    rawCapture.truncate(0)

    if fv < set_point:
        stepper.run(1000, 1)
    else:
        stepper.stop()
        camera.close()
