#!/usr/bin/python3
"""
Deploy this script on the robot. The robot will play and collect experience data according to its imagination.
"""
import sys
import os
import time
import numpy as np
import cv2
import tensorflow as tf
from datetime import datetime
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)

from drivers.mecanum_driver import MecanumDriver
from agents.ima_macromphalus import OnPolicyBuffer, IntrinsicMotivationAgent

# Parameters
version = 'macromphalus'
dim_view = (128,128,1)
dim_latent = 8
num_act = 10
dim_act = 1
total_steps = 300
time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
image_dir = os.path.join('/ssd', version+'_experience', time_stamp, 'views')
if not os.path.exists(image_dir):
    try:
        os.makedirs(image_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
# Get camera, motor driver, controller and replay buffer ready 
# eye = cv2.VideoCapture(0) # usb webcam for debugging
eye = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)128, height=(int)128, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)
eye.get(cv2.CAP_PROP_FPS)
wheels = MecanumDriver() # need integrate mecdriver into agent in next version
brain = IntrinsicMotivationAgent(dim_view=dim_view, dim_latent=dim_latent, num_act=num_act, dim_act=dim_act)
memory = OnPolicyBuffer(max_size=total_steps)
# Load models
model_dir = os.path.join(sys.path[0], 'model_dir', version, '2021-04-13-17-29')
brain = IntrinsicMotivationAgent(dim_view=dim_view, dim_latent=dim_latent, num_act=num_act, dim_act=dim_act)
brain.vae.encoder.encoder_net = tf.keras.models.load_model(os.path.join(model_dir, 'encoder'))
brain.vae.decoder.decoder_net = tf.keras.models.load_model(os.path.join(model_dir, 'decoder'))
brain.imaginator.dynamics_net = tf.keras.models.load_model(os.path.join(model_dir, 'imaginator'))
brain.ac.actor.policy_net = tf.keras.models.load_model(os.path.join(model_dir, 'actor'))
brain.ac.critic.value_net = tf.keras.models.load_model(os.path.join(model_dir, 'critic'))

# Ignition parameters
frame_counter = 0
step_counter = 0
time_elapse = 0
ret = 0
stepwise_frame_ids = []
action_data = np.zeros(shape=(total_steps,), dtype=np.float32)
# Start playing
start_time = time.time()
ret, frame = eye.read()
view = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255. # from 0~255 to 0~1
view.resize(1,128,128,1)
mu, logsigma = brain.vae.encoder(view)
obs = brain.vae.reparameterize(mu, logsigma)
act, val, lpa = brain.ac.make_decision(tf.expand_dims(obs,0))
imn_mu, imn_logsigma = brain.imaginator(tf.expand_dims(obs,0), tf.reshape(act,(1,1)))
imn_obs = brain.vae.reparameterize(imn_mu, imn_logsigma)
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
            ret += rew
            memory.store(
                obs, act, rew, val, lpa, nxt_obs, imn_obs, mu, logsigma, nxt_mu, nxt_logsigma, imn_mu, imn_logsigma
            )
            logging.debug("\nstep: {} \ncurrent state: {} \naction: {} \nimagination: {} \nnext state: {} \nvalue: {} \nlog prob: {} \nreward: {}".format(step_counter, obs, act, imn_obs, nxt_obs, val, lpa, rew))
            stepwise_frame_ids.append(frame_counter)
            step_counter += 1
            logging.info("\n----\nStep: {} \nTotalFrames: {} \nReturn: {} \n----\n".format(step_counter, frame_counter, ret))
            obs = nxt_obs
            mu = nxt_mu
            logsigma = nxt_logsigma
            act, val, lpa = brain.ac.make_decision(tf.expand_dims(obs,0))
            imn_mu, imn_logsigma = brain.imaginator(tf.expand_dims(obs,0), tf.reshape(act,(1,1)))
            imn_obs = brain.vae.reparameterize(imn_mu, imn_logsigma)
            wheels.set_action(act)
except KeyboardInterrupt:    
    print("\r\nctrl + c:")
    eye.release()
    cv2.destroyAllWindows()
    wheels.halt()
    exit()
# Finish memory
_, val, _ = brain.make_decision(tf.expand_dims(obs,0))
memory.finish_path(val)

# Save valuable items
with open(os.path.join(os.path.dirname(image_dir), 'elapsed_time.txt'), 'w') as f:
    f.write("{}".format(time.time()-start_time))
replay_data = memory.get()
np.save(os.path.join(os.path.dirname(image_dir), 'replay_data.npy'), replay_data)
np.save(os.path.join(os.path.dirname(image_dir), 'stepwise_frame_ids.npy'), stepwise_frame_ids)
np.save(os.path.join(os.path.dirname(image_dir), 'return.npy'), ret)
# When everything done, release the capture and stop motors
eye.release()
cv2.destroyAllWindows()
wheels.halt()

