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

from drivers.mecanum_driver import MecanumDriver
from agents.ima_macromphalus import IntrinsicMotivationAgent, OnPolicyBuffer


# Parameters
versions = ['belauensis', 'macromphalus', 'pompilius', 'scrobiculatus', 'stenomphalus'] # In memory of the vanished pattern in front of the Rhodes Hall. Every new born agent will be given a name after a valid species in the family of Nautilidae listed in https://en.wikipedia.org/wiki/Nautilus_(genus)
dim_view = (128,128,1)
dim_latent = 8
num_act = 10
dim_act = 1
total_steps = 300
time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
image_dir = os.path.join('/ssd', versions[1]+'_experience', time_stamp, 'views')
if not os.path.exists(image_dir):
    try:
        os.makedirs(image_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
# Instantiate camera, motor driver, agent, buffer
eye = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)128, height=(int)128, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)
eye.get(cv2.CAP_PROP_FPS)
wheels = MecanumDriver() # need integrate mecdriver into agent in next version
brain = IntrinsicMotivationAgent(dim_view=dim_view, dim_latent=dim_latent, num_act=num_act, dim_act=dim_act)
memory = OnPolicyBuffer(max_size=total_steps)
# Ignition parameters
frame_counter = 0
step_counter = 0
time_elapse = 0
stepwise_frame_ids = []
action_data = np.zeros(shape=(total_steps,), dtype=np.float32)

# Start data collection
start_time = time.time()
ret, frame = eye.read()
view = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255. # from 0~255 to 0~1
view.resize(1,128,128,1)
mu, logsigma = brain.vae.encoder(img)
obs = brain.vae.reparameterize(mu, logsigma)
act = np.random.randint(low=0, high=num_act)
imn_mu, imn_logsigma = brain.imaginator(tf.expand_dims(obs,0), tf.reshape(act,(1,1)))
imn_obs = brain.imaginator.reparameterize(imn_mu, imn_logsigma)
wheels.set_action(act)
try:
    while step_counter < total_steps:
        prev_time_elapse = time_elapse
        time_elapse = time.time() - start_time
        ret, frame = eye.read()
        cv2.imwrite(os.path.join(image_dir, str(frame_counter)+'.jpg'), frame)
        frame_counter+=1
        if int(time_elapse)-int(prev_time_elapse): # take a random action in every sec
            nxt_view = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255.
            nxt_view.resize(1,128,128,1)
            nxt_mu, nxt_logsigma = brain.vae.encoder(nxt_view)
            nxt_obs = brain.vae.reparameterize(nxt_mu, nxt_logsigma)
            rew = brain.compute_intrinsic_reward(nxt_mu, nxt_logsigma, imn_obs)
            memory.store(
                obs, act, rew, 0, 0, nxt_obs, imn_obs, mu, logsigma, nxt_mu, nxt_logsigma, imn_mu, imn_logsigma
            )
            stepwise_frame_ids.append(frame_counter)
            step_counter += 1
            obs = nxt_obs
            mu = nxt_mu
            logsigma = nxt_logsigma
            act = np.random.randint(low=0, high=num_act)
            imn_mu, imn_logsigma = brain.imaginator(tf.expand_dims(obs,0), tf.reshape(act,(1,1)))
            imn_obs = brain.imaginator.reparameterize(imn_mu, imn_logsigma)
            wheels.set_action(act)
except KeyboardInterrupt:    
    print("\r\nctrl + c:")
    eye.release()
    cv2.destroyAllWindows()
    wheels.halt()
    exit()
# Save valuable items
with open(os.path.join(os.path.dirname(image_dir), 'elapsed_time.txt'), 'w') as f:
    f.write("{}".format(time.time()-start_time))
np.save(os.path.join(os.path.dirname(image_dir), 'action_data.npy'), action_data)
np.save(os.path.join(os.path.dirname(image_dir), 'stepwise_frames.npy'), stepwise_frames)
# When everything done, release the capture and stop motors
eye.release()
cv2.destroyAllWindows()
wheels.halt()

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


