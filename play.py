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
total_steps = 300
dim_latent = 16
dim_view = (128,128,1)
dim_act = 1
num_act = 10

# Get mecanum driver ready
wheels = MecanumDriver() # need integrate mecdriver into agent in next version
# Get camera ready
# eye = cv2.VideoCapture(0) # usb webcam for debugging
eye = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)128, height=(int)128, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)
eye.get(cv2.CAP_PROP_FPS)
# Get agent ready
brain = IntrinsicMotivationAgent(dim_latent=dim_latent, dim_view=dim_view, dim_act=dim_act, num_act=num_act)
seed = 'belauensis'
load_dir = os.path.join(sys.path[0], 'model_dir', seed, '2021-01-08-19-45') # typically use the last saved models
brain.encoder = tf.keras.models.load_model(os.path.join(load_dir, 'encoder'))
brain.decoder = tf.keras.models.load_model(os.path.join(load_dir, 'decoder'))
brain.imaginator = tf.keras.models.load_model(os.path.join(load_dir, 'imaginator'))
brain.actor = tf.keras.models.load_model(os.path.join(load_dir, 'actor'))
brain.critic = tf.keras.models.load_model(os.path.join(load_dir, 'critic'))
memory = OnPolicyBuffer(dim_latent=dim_latent, dim_act=dim_act, size=total_steps, gamma=.99, lam=.97)
# Generate first imagination and action
ret, frame = eye.read() # obs = env.reset()
view = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255. # from 0~255 to 0~1
view.resize(1,128,128,1)
latent_distribution = brain.encode(view) # encoded distribution
state = latent_distribution.sample() # brain.reparameterize(latent_distribution.mean(), latent_distribution.stddev())
act, val, logp = brain.make_decision(state) 
wheels.set_action(int(act))
imagination = brain.imagine(state, np.reshape(act, (1,1)).astype(np.float32)) # imagined latent state
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
            next_view = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255.
            next_view.resize(1,128,128,1)
            next_latent_distribution = brain.encode(next_view)
            next_state = next_latent_distribution.sample() # brain.reparameterize(next_latent_distribution.mean(), next_latent_distribution.stddev()) 
            rew = brain.compute_intrinsic_reward(imagination, next_latent_distribution)
            ep_ret+=rew
            ep_len+=1
            memory.store(
                np.squeeze(latent_distribution.mean()), 
                np.squeeze(latent_distribution.stddev()), 
                np.squeeze(next_latent_distribution.mean()), 
                np.squeeze(next_latent_distribution.stddev()), 
                np.squeeze(state), 
                np.squeeze(next_state), 
                np.squeeze(imagination), 
                act, 
                rew, 
                val, 
                logp
            )
            step_counter+=1
            stepwise_frames.append(frame_counter)
            logging.info("\nepisode: {} \nstep: {} \ncurrent state: {} \nimagination: {} \naction: {} \nnext state: {} \nvalue: {} \nlog prob: {} \nreward: {} \nepisode return: {} \nepisode length: {}".format(episode_counter+1, step_counter, state, imagination, act, next_state, val, logp, rew, ep_ret, ep_len))
            # handle episode terminal
            if not step_counter%max_ep_len:
                _, val, _ = brain.make_decision(state)
                memory.finish_path(np.squeeze(val))
                episode_counter+=1
                episodic_returns.append(ep_ret)
                logging.info("\n----\nTotalFrames: {} \nEpisode: {}, EpReturn: {} \n----\n".format(frame_counter, episode_counter, ep_ret))
                ep_ret, ep_len = 0, 0
            # compute next obs, act, val, logp
            state = next_state # SUPER CRITICAL!!!
            act, val, logp = brain.make_decision(state) 
            wheels.set_action(int(act))
            imagination = brain.imagine(state, np.reshape(act, (1,1)).astype(np.float32))
            
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

