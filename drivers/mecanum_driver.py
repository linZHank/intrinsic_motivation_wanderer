#!/usr/bin/env python3
"""
A simple class for driving mecanum_wheel bot with discrete action space
"""

import os
import time
import numpy as np
import cv2
from motor_driver import MotorDriver

class MecanumDriver:

    def __init__(self, motor_speed=25):
        self.fw_driver = MotorDriver(driver_addr=0x40)
        self.rw_driver = MotorDriver(driver_addr=0x50)
        self.action_options = [
            'f',
            'b',
            'l',
            'r',
            'lf',
            'lb',
            'rf',
            'rb',
            'c',
            'cc',
        ]
        self.motor_speed = motor_speed # 0~100
        self.cap = cv1.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)128, height=(int)128, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)

    def halt(self):
        self.fw_driver.stop_motor(motor_index=0)
        self.fw_driver.stop_motor(motor_index=1)
        self.rw_driver.stop_motor(motor_index=0)
        self.rw_driver.stop_motor(motor_index=1)
        cap.release()
        cv2.destroyAllWindows()
        print("\nSTOPPED\n")
        
    def set_action(self, action):
        assert 0<=action<len(self.action_options)
        if action==0:
            # translational move forward
            self.fw_driver.run_motor(motor_index=0, direction='forward', speed=self.motor_speed)
            self.fw_driver.run_motor(motor_index=1, direction='forward', speed=self.motor_speed)
            self.rw_driver.run_motor(motor_index=0, direction='forward', speed=self.motor_speed)
            self.rw_driver.run_motor(motor_index=1, direction='forward', speed=self.motor_speed)
            print("\nmoving forward\n")

        if action==1:
            # translational move backward
            self.fw_driver.run_motor(motor_index=0, direction='backward', speed=self.motor_speed)
            self.fw_driver.run_motor(motor_index=1, direction='backward', speed=self.motor_speed)
            self.rw_driver.run_motor(motor_index=0, direction='backward', speed=self.motor_speed)
            self.rw_driver.run_motor(motor_index=1, direction='backward', speed=self.motor_speed)
            print("\nmoving backward\n")

        if action==2:
            # translational move leftward
            self.fw_driver.run_motor(motor_index=0, direction='backward', speed=self.motor_speed)
            self.fw_driver.run_motor(motor_index=1, direction='forward', speed=self.motor_speed)
            self.rw_driver.run_motor(motor_index=0, direction='forward', speed=self.motor_speed)
            self.rw_driver.run_motor(motor_index=1, direction='backward', speed=self.motor_speed)
            print("\nmoving leftward\n")

        if action==3:
            # translational move rightward
            self.fw_driver.run_motor(motor_index=0, direction='forward', speed=self.motor_speed)
            self.fw_driver.run_motor(motor_index=1, direction='backward', speed=self.motor_speed)
            self.rw_driver.run_motor(motor_index=0, direction='backward', speed=self.motor_speed)
            self.rw_driver.run_motor(motor_index=1, direction='forward', speed=self.motor_speed)
            print("\nmoving rightward\n")

        if action==4:
            # translational move left-forward
            self.fw_driver.run_motor(motor_index=1, direction='forward', speed=self.motor_speed)
            self.rw_driver.run_motor(motor_index=0, direction='forward', speed=self.motor_speed)
            print("\nmoving left-forward\n")

        if action==5:
            # translational move right-forward
            self.fw_driver.run_motor(motor_index=0, direction='forward', speed=self.motor_speed)
            self.rw_driver.run_motor(motor_index=1, direction='forward', speed=self.motor_speed)
            print("\nmoving right-forward\n")

        if action==6:
            # translational move left-backward
            self.fw_driver.run_motor(motor_index=0, direction='backward', speed=self.motor_speed)
            self.rw_driver.run_motor(motor_index=1, direction='backward', speed=self.motor_speed)
            print("\nmoving left-backward\n")

        if action==7:
            # translational move right-backward
            self.fw_driver.run_motor(motor_index=1, direction='backward', speed=self.motor_speed)
            self.rw_driver.run_motor(motor_index=0, direction='backward', speed=self.motor_speed)
            print("\nmoving right-backward\n")

        if action==8:
            # clockwise rotation
            self.fw_driver.run_motor(motor_index=0, direction='forward', speed=self.motor_speed)
            self.fw_driver.run_motor(motor_index=1, direction='backward', speed=self.motor_speed)
            self.rw_driver.run_motor(motor_index=0, direction='forward', speed=self.motor_speed)
            self.rw_driver.run_motor(motor_index=1, direction='backward', speed=self.motor_speed)
            print("\nrotation clockwise\n")

        if action==9:
            # counter-clockwise rotation
            self.fw_driver.run_motor(motor_index=0, direction='backward', speed=self.motor_speed)
            self.fw_driver.run_motor(motor_index=1, direction='forward', speed=self.motor_speed)
            self.rw_driver.run_motor(motor_index=0, direction='backward', speed=self.motor_speed)
            self.rw_driver.run_motor(motor_index=1, direction='forward', speed=self.motor_speed)
            print("\nrotating counter-clockwise\n")

if __name__=='__main__':
    mec = MecanumDriver()
    for i in range(len(mec.action_options)):
        ret, frame = mec.cap.read()
        obs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame, comment next 3 lines out if headless
        cv2.imshow('frame', obs)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            mec.halt()
            break
        mec.set_action(i)
        time.sleep(4)
        mec.halt()
