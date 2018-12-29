# PowderScale
This is a Raspberry Pi powered powder scale dispenser. It connects with
Adafruit IO and when a new powder weight is pushed into a feed, the system
wakes up, and dispenses powder.

This system uses the Pi camera to read the LCD display of a GS-1500 powder
scale. It then drives an ST L6472 stepper motor driver to drive a stepper motor
I found in my box of motors. I then 3D printed a bunch of stuff to hold the Pi,
Pi camera, stepper motor, and stepper motor eval board.

The stepper motor spins a Frankford Arsenal trickle powderer.

This repo contains the python code for the LCD Pi camera character recongition
system and drives the stepper motor.