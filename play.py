#!/usr/bin/python3
"""
Deploy this script on the robot. The robot will play and collect experience data according to its imagination.
"""
import sys
import os
import time
import numpy as np
import cv2
from datetime import datetime
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

from drivers.mecanum_driver import MecanumDriver
from agents.intrinsic_motivation_agent import OnPolicyBuffer, IntrinsicMotivationAgent
import tensorflow as tf

# Parameters
total_steps = 1000
dim_latent = 16
dim_view = (128,128,1)
num_act = 10
dim_act = 1

# Get mecanum driver ready
wheels = MecanumDriver() # need integrate mecdriver into agent in next version
# Get camera ready
# eye = cv2.VideoCapture(0) # usb webcam for debugging
eye = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)128, height=(int)128, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)
eye.get(cv2.CAP_PROP_FPS)
# Get agent ready
brain = IntrinsicMotivationAgent(dim_view, dim_latent, num_act, dim_act)
seed = 'macromphalus'
load_dir = os.path.join(sys.path[0], 'model_dir', seed, '2021-03') # typically use the last saved models
brain.vae.encoder = tf.keras.models.load_model(os.path.join(load_dir, 'encoder'))
brain.vae.decoder = tf.keras.models.load_model(os.path.join(load_dir, 'decoder'))
brain.imaginator = tf.keras.models.load_model(os.path.join(load_dir, 'imaginator'))
brain.actor = tf.keras.models.load_model(os.path.join(load_dir, 'actor'))
brain.critic = tf.keras.models.load_model(os.path.join(load_dir, 'critic'))
memory = OnPolicyBuffer(max_size=total_steps)
# Generate first imagination and action
ret, frame = eye.read() # obs = env.reset()
view = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255. # from 0~255 to 0~1
view.resize(1,128,128,1)
obs, _ = brain.vae.encode(view) # encoded distribution
act, val, lpa = brain.ac.make_decision(obs) 
mu_imn, logsigma_imn = brain.imaginator(obs, np.reshape(act, (1,1))) # imagined latent state
wheels.set_action(int(act))
logging.info("\n====Ignition====\n")
# Preapare for experience collecting
save_dir = '/ssd/mecanum_experience/' + datetime.now().strftime("%Y-%m-%d-%H-%M") + '/'
views_dir = save_dir+'views'
if not os.path.exists(views_dir):
    try:
        os.makedirs(views_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
max_ep_len = 10
ep_ret, ep_len = 0, 0
frame_counter = 0
episode_counter = 0
step_counter = 0
episodic_returns = []
stepwise_frames = []
time_elapse = 0
prev_time_elapse = 0
start_time = time.time()

# Main loop
try:
    while step_counter < total_steps:
        ret, frame = eye.read()
        cv2.imwrite(os.path.join(views_dir, str(frame_counter)+'.jpg'), frame)
        prev_time_elapse = time_elapse
        time_elapse = time.time() - start_time
        frame_counter+=1
        if int(time_elapse)-int(prev_time_elapse): # take an action every sec
            view_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255.
            view_next.resize(1,128,128,1)
            mu, logsigma = brain.vae.encoder(view_next)
            rew = brain.compute_intrinsic_reward(mu_imn, logsigma_imn, mu)
            ep_ret+=rew
            ep_len+=1
            memory.store(obs, act, rew, val, lpa, mu, logsigma)
            step_counter+=1
            stepwise_frames.append(frame_counter)
            logging.debug("\nepisode: {} \nstep: {} \ncurrent state: {} \nimagination: {} \naction: {} \nnext state: {} \nvalue: {} \nlog prob: {} \nreward: {} \nepisode return: {} \nepisode length: {}".format(episode_counter+1, step_counter, obs, (mu_imn, logsigma_imn), act, (mu, logsigma), val, logp, rew, ep_ret, ep_len))
            # handle episode terminal
            if not step_counter%max_ep_len:
                _, val, _ = brain.make_decision(obs)
                memory.finish_path(val)
                episode_counter+=1
                episodic_returns.append(ep_ret)
                logging.info("\n----\nTotalFrames: {} \nEpisode: {}, EpReturn: {} \n----\n".format(frame_counter, episode_counter, ep_ret))
                ep_ret, ep_len = 0, 0
            # compute next obs, act, val, logp
            obs = mu # SUPER CRITICAL!!!
            act, val, logp = brain.make_decision(obs) 
            wheels.set_action(int(act))
            mu_imn, logsigma_imn = brain.imaginator(obs, np.reshape(act, (1,1)))
            
    # Save valuable items
    with open(os.path.join(save_dir, 'elapsed_time.txt'), 'w') as f:
        f.write("{}".format(time.time()-start_time))
    replay_data = memory.get()
    np.save(os.path.join(save_dir, 'replay_data.npy'), replay_data)
    np.save(os.path.join(save_dir, 'stepwise_frames.npy'), stepwise_frames)
    np.save(os.path.join(save_dir, 'episodic_returns.npy'), episodic_returns)
except KeyboardInterrupt:    
    print("\r\nctrl + c:")
    eye.release()
    cv2.destroyAllWindows()
    wheels.halt()
    exit()

# When everything done, release the capture and stop motors
eye.release()
cv2.destroyAllWindows()
wheels.halt()

