#!/usr/bin/python3

from PCA9685 import PCA9685
import time


DIR_LIST = ['forward', 'backward']

class MotorDriver:

    def __init__(self, driver_addr=0x40, freq=50):
        self.driver = PCA9685(driver_addr, debug=True)
        self.driver.setPWMFreq(freq)
        self.PWMA = 0
        self.AIN1 = 1
        self.AIN2 = 2
        self.PWMB = 5
        self.BIN1 = 3
        self.BIN2 = 4

    def run_motor(self, motor_index, direction, speed):
        # syntax check
        assert direction in DIR_LIST
        if speed>100:
            speed = 100
        # run motor
        # wire connection: red->1, white->2
        if motor_index==0:
            self.driver.setDutycycle(self.PWMA, speed)
            if direction==DIR_LIST[0]:
                self.driver.setLevel(self.AIN1, 1)
                self.driver.setLevel(self.AIN2, 0)
            elif direction==DIR_LIST[1]:
                self.driver.setLevel(self.AIN1, 0)
                self.driver.setLevel(self.AIN2, 1)
        elif motor_index==1: 
            self.driver.setDutycycle(self.PWMB, speed)
            if direction==DIR_LIST[0]:
                self.driver.setLevel(self.BIN1, 0)
                self.driver.setLevel(self.BIN2, 1)
            elif direction==DIR_LIST[1]:
                self.driver.setLevel(self.BIN1, 1)
                self.driver.setLevel(self.BIN2, 0)
        else:
            print("Cannot run Motor: {}. Use 0 or 1 as the motor_index")

    def stop_motor(self, motor_index):
        if motor_index==0:
            self.driver.setDutycycle(self.PWMA, 0)
        elif motor_index==1:
            self.driver.setDutycycle(self.PWMB, 0)
        else:
            print("Cannot stop Motor: {}. Use 0 or 1 as the motor_index")


