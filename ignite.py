#!/usr/bin/python3
"""
Initialize Intrinsic Motivation Agent (IMA)
"""
import sys
import os
from datetime import datetime
import time
import numpy as np
import cv2
import tensorflow as tf
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

from agents.ima_macromphalus import IntrinsicMotivationAgent


# Parameters
versions = ['belauensis', 'macromphalus', 'pompilius', 'scrobiculatus', 'stenomphalus'] # For memorization of the vanished pattern in front of the Rhodes Hall. Every new born agent will be given a name after a valid species in the family of Nautilidae listed in https://en.wikipedia.org/wiki/Nautilus_(genus)
dim_view = (128,128,1)
dim_latent = 8
num_act = 10
dim_act = 1
time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
# Instantiate camera, motor driver, agent
eye = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)128, height=(int)128, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)
eye.get(cv2.CAP_PROP_FPS)
image_dir = os.path.join('/ssd', version+'_experience', time_stamp, 'views')
if not os.path.exists(image_dir):
    try:
        os.makedirs(image_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
wheels = MecanumDriver() # need integrate mecdriver into agent in next version
brain = IntrinsicMotivationAgent(dim_view, dim_latent, num_act, dim_act)
# Data collection
start_time = time.time()
try:
    while step_counter < 10:
        ret, frame = eye.read()
        cv2.imwrite(os.path.join(image_dir, str(frame_counter)+'.jpg'), frame)
        prev_time_elapse = time_elapse
        time_elapse = time.time() - start_time
        frame_counter+=1
        if int(time_elapse)-int(prev_time_elapse): # take an action every sec
            wheels.set_action(int(np.random.randint(low=0, high=num_act)))
except KeyboardInterrupt:    
    print("\r\nctrl + c:")
    eye.release()
    cv2.destroyAllWindows()
    wheels.halt()
    exit()

# Set model paths
model_dir = os.path.join(sys.path[0], 'model_dir', versions[1], time_stamp)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
encoder_path = os.path.join(model_dir, 'encoder')
decoder_path = os.path.join(model_dir, 'decoder')
imaginator_path = os.path.join(model_dir, 'imaginator')
actor_path = os.path.join(model_dir, 'actor')
critic_path = os.path.join(model_dir, 'critic')
# Save models
brain.vae.encoder.encoder_net.save(encoder_path)
brain.vae.decoder.decoder_net.save(decoder_path)
brain.imaginator.dynamics_net.save(imaginator_path)
brain.ac.actor.policy_net.save(actor_path)
brain.ac.critic.value_net.save(critic_path)
