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

s = ScaleOGR()


camera = PiCamera()
rawCapture = PiRGBArray(camera)

while True:
    camera.capture(rawCapture, format="bgr")
    image = rawCapture.array
    num = s.process(image)
    print('Detected number: {:}'.format(num))
    time.sleep(0.1)
