#!/usr/bin/python3
"""
A script to test mecanum driver
"""
import time
import numpy as np
from drivers.mecanum_driver import MecanumDriver

wheels = MecanumDriver()
try:
    while True:
        act = np.random.randint(0,10)
        wheels.set_action(int(act))
        time.sleep(2)
except KeyboardInterrupt:    
    print("\r\nctrl + c:")
    wheels.halt()
    exit()

# When everything done, release the capture and stop motors
wheels.halt()

