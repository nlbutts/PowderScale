#!/usr/bin/python3

import sys

# Import Adafruit IO MQTT client.
from Adafruit_IO import MQTTClient

from scaleogr import ScaleOGR
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
from stepper import Stepper
from adafruit_secrets import *

class Filter():
    def __init__(self, gain):
        self.gain = gain
        self.filtered_value = 0

    def filter(self, value):
        err = value - self.filtered_value
        newV = err * self.gain
        self.filtered_value += newV

        return self.filtered_value



# Set to the ID of the feed to subscribe to for updates.
FEED_ID = 'Powder'


# Define callback functions which will be called when certain events happen.
def connected(client):
    # Connected function will be called when the client is connected to Adafruit IO.
    # This is a good place to subscribe to feed changes.  The client parameter
    # passed to this function is the Adafruit IO MQTT client so you can make
    # calls against it easily.
    print('Connected to Adafruit IO!  Listening for {0} changes...'.format(FEED_ID))
    # Subscribe to changes on a feed named DemoFeed.
    client.subscribe(FEED_ID)

def disconnected(client):
    # Disconnected function will be called when the client disconnects.
    print('Disconnected from Adafruit IO!')
    sys.exit(1)

def message(client, feed_id, payload):
    # Message function will be called when a subscribed feed has a new value.
    # The feed_id parameter identifies the feed, and the payload parameter has
    # the new value.
    print('Feed {0} received new value: {1}'.format(feed_id, payload))
    dispense(payload)

def dispense(value):
    value = float(value)
    try:
        print("Setting up OGR")
        ogr = ScaleOGR(False, False)
        stepper = Stepper(1000000)
        f = Filter(0.2)

        print("Setting up camera")
        camera = PiCamera()
        camera.close()
        camera = PiCamera()
        camera.resolution = (1920,1088)
        camera.framerate = 30
        rawCapture = PiRGBArray(camera, size=(1920,1088))

        print("Getting images")
        for frame in camera.capture_continuous(rawCapture, format="bgr",  use_video_port=True):
            image = rawCapture.array
            num = ogr.process(image)
            fv = f.filter(num)
            print('Detected number: {:} -- Filtered number: {:}'.format(num, fv))
            rawCapture.truncate(0)

            if fv < value:
                stepper.run(1000, 1)
            else:
                stepper.stop()
                camera.close()

        pass
    except Exception as e:
        print(e)
        print("The scale is probably not on, turn it on and try again")
        pass


# Create an MQTT client instance.
client = MQTTClient(ADAFRUIT_IO_USERNAME, ADAFRUIT_IO_KEY)

# Setup the callback functions defined above.
client.on_connect    = connected
client.on_disconnect = disconnected
client.on_message    = message

# Connect to the Adafruit IO server.
client.connect()

# Start a message loop that blocks forever waiting for MQTT messages to be
# received.  Note there are other options for running the event loop like doing
# so in a background thread--see the mqtt_client.py example to learn more.
client.loop_blocking()