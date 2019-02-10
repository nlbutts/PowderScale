#!/usr/bin/python3

from scaleogr import ScaleOGR
import cv2
import time
import matplotlib.pyplot as plt

class Filter():
    def __init__(self, gain):
        self.gain = gain
        self.filtered_value = 0

    def filter(self, value):
        err = value - self.filtered_value
        newV = err * self.gain
        self.filtered_value += newV

        return self.filtered_value


def run(ogr, f):
    raw = []
    filtered = []
    count = 0
    print("Testing")
    cap = cv2.VideoCapture(r'C:\ztemp\test.mp4')
    #cap = cv2.VideoCapture('../data/test.mp4')
    if cap.isOpened():
        ret, img = cap.read()
        while ret:
            value = ogr.process(img)
            fvalue = f.filter(value)
            raw.append(value)
            filtered.append(fvalue)            
            #time.sleep(0.1)
            print('Raw: {:} -- filtered: {:}'.format(value, fvalue))
            ret, img = cap.read()


        plt.clf()
        plt.plot(raw, label='Raw')
        plt.plot(filtered, label='Filtered')
        plt.show()

    else:
        print("Test file not found")

    return raw, filtered

s = ScaleOGR(False)
f = Filter(0.2)


#s.run_training(r'c:\ztemp\test.mp4')
raw, filtered = run(s, f)
