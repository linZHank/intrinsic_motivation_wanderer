#!/usr/bin/python3
"""
Deploy this script on the robot. The robot will play and collect experience data according to its imagination.
"""
import os
import time
import numpy as np
import cv2
from datetime import datetime

from drivers.mecanum_driver import MecanumDriver
from agents.intrinsic_motivation_agent import OnPolicyBuffer, IntrinsicMotivationAgent

# Instantiate mecanum driver
wanderer = MecanumDriver() # need integrate mecdriver into agent in next version
# Get camera ready
eye = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)128, height=(int)128, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)
# eye = cv2.VideoCapture(0) # usb webcam debug
ret, frame = eye.read()
img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255. # from 0~255 to 0~1
img.resize(1,128,128,1)
# Set experience saving dir
save_dir = '/ssd/mecanum_experience/' + datetime.now().strftime("%Y-%m-%d-%H-%M") + '/'
if not os.path.exists(save_dir):
    try:
        os.makedirs(save_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
# Parameters
total_steps = 30
max_ep_len = 10
dim_latent = 8
dim_origin=(128,128,1)
dim_obs=(128,128,2)
dim_act=1
num_act = 10
# Get agent ready
brain = IntrinsicMotivationAgent(dim_latent=dim_latent, dim_origin=dim_origin, act_type='discrete', dim_obs=dim_obs, dim_act=dim_act, num_act=num_act)
brain.imagine(img) 
print("\n====Reset====\nencoded state: {} \nimagined state: {}\n".format(brain.autoencoder.encode(img), (brain.imagination.mean(), brain.imagination.stddev())))
memory = OnPolicyBuffer(dim_act=dim_act, size=total_steps, gamma=.99, lam=.97)
obs = np.concatenate((img, brain.decoded_imagination), axis=-1)
act, val, logp = brain.pi_of_a_given_s(obs) 
wanderer.set_action(int(act))
# Preapare for experience collecting
ep_rew, ep_len = 0, 0
episode_counter = 0
obs_counter = 0
step_counter = 0
buffer_actions = []
buffer_obs_counters = []
time_elapse = 0
prev_time_elapse = 0
start_time = time.time()
# Main loop
try:
    for st in range(total_steps):
        ret, frame = eye.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255.
        cv2.imwrite(os.path.join(save_dir, str(frame_counter)+'.jpg'), frame)
        prev_time_elapse = time_elapse
        time_elapse = time.time() - start_time
        obs_counter+=1
        if not int(time_elapse)%2 and int(prev_time_elapse)%2: # change mecanum's behavior every 2 sec
            obs = np.concatenate((img, brain.decoded_imagination), axis=-1)
            rew = brain.compute_intrinsic_reward(np.expand_dims(img)
            ep_ret+=rew
            ep_len+=1
            memory.store(act, rew, val, logp)
            act, val, logp = brain.pi_of_a_given_s(obs) 
            wanderer.set_action(int(act))
            print("\nstep: {} \nencoded state: {} \n".format(step_counter+1, brain.autoencoder.encode(np.expand_dims(np.expand_dims(obs, -1), 0))))
            step_counter+=1
            buffer_obs_counters.append(obs_counter)
            # handle episode terminal
            if not step_counter%max_ep_len:
                _, val, _ = brain.pi_of_a_given_s(obs)
                memory.finish_path(np.squeeze(val))
                episode_counter+=1
                episodic_returns.append(ep_ret)
                sedimentary_returns.append(sum(episodic_returns)/episode_counter)
                # reset
                brain.imagine(np.expand_dims(np.expand_dims(obs, -1), 0))
            
    replay_data = replay_buffer.get()
    np.save(os.path.join(save_dir, 'replay_buffer.npy'), replay_data)
    np.save(os.path.join(save_dir, 'buffer_obs_counters.npy'), buffer_obs_counters)
except KeyboardInterrupt:    
    print("\r\nctrl + c:")
    eye.release()
    cv2.destroyAllWindows()
    wanderer.halt()
    exit()

# When everything done, release the capture and stop motors
eye.release()
cv2.destroyAllWindows()
wanderer.halt()

