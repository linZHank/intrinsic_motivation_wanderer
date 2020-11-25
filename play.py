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
memory = OnPolicyBuffer(dim_obs=dim_obs, dim_latent=dim_latent, dim_act=dim_act, size=total_steps, gamma=.99, lam=.97)
# Get mecanum driver ready
wheels = MecanumDriver() # need integrate mecdriver into agent in next version
# Get camera ready
eye = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)128, height=(int)128, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)
# eye = cv2.VideoCapture(0) # usb webcam for debugging
ret, frame = eye.read()
img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255. # from 0~255 to 0~1
img.resize(1,128,128,1)
# Generate first imagination and action
brain.imagine(img) 
print("\n====Reset====\nencoded state: {} \nimagined state: {}\n".format(brain.encoder(img), (brain.imagination.mean(), brain.imagination.stddev())))
obs = np.concatenate((img, brain.decoded_imagination), axis=-1)
act, val, logp = brain.pi_of_a_given_s(obs) 
wheels.set_action(int(act))
# Preapare for experience collecting
save_dir = '/ssd/mecanum_experience/' + datetime.now().strftime("%Y-%m-%d-%H-%M") + '/'
if not os.path.exists(save_dir):
    try:
        os.makedirs(save_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
ep_ret, ep_len = 0, 0
frame_counter = 0
episode_counter = 0
step_counter = 0
episodic_returns, sedimentary_returns = [], []
stepwise_frames = []
time_elapse = 0
prev_time_elapse = 0
start_time = time.time()
# Main loop
try:
    while step_counter < total_steps:
        ret, frame = eye.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255.
        img.resize(1,128,128,1)
        cv2.imwrite(os.path.join(save_dir, str(frame_counter)+'.jpg'), frame)
        prev_time_elapse = time_elapse
        time_elapse = time.time() - start_time
        frame_counter+=1
        if not int(time_elapse)%2 and int(prev_time_elapse)%2: # change mecanum's behavior every 2 sec
            rew = brain.compute_intrinsic_reward(img)
            ep_ret+=rew
            ep_len+=1
            memory.store(obs, np.squeeze(brain.imagination), act, rew, val, logp)
            print("\nstep: {} \nencoded state: {} \naction: {} \nvalue: {} \nlog prob: {} \nreward: {} \nepisode return: {} \n episode length".format(step_counter+1, brain.encoder(img), act, val, logp, rew, ep_ret, ep_len))
            step_counter+=1
            stepwise_frames.append(frame_counter)
            # handle episode terminal
            if not step_counter%max_ep_len:
                _, val, _ = brain.pi_of_a_given_s(obs)
                memory.finish_path(np.squeeze(val))
                episode_counter+=1
                episodic_returns.append(ep_ret)
                sedimentary_returns.append(sum(episodic_returns)/episode_counter)
                print("\n----\nTotalFrames: {} \nEpisode: {}, EpReturn: {}, EpLength: {} \n----\n".format(frame_counter, episode_counter, ep_ret, ep_len))
                # reset
                brain.imagine(img)
            # compute next obs, act, val, logp
            obs = np.concatenate((img, brain.decoded_imagination), axis=-1)
            act, val, logp = brain.pi_of_a_given_s(obs) 
            wheels.set_action(int(act))
            
    # Save valuable items
    replay_buffer = memory.get()
    np.save(os.path.join(save_dir, 'replay_buffer.npy'), replay_buffer)
    np.save(os.path.join(save_dir, 'stepwise_frames.npy'), stepwise_frames)
    np.save(os.path.join(save_dir, 'episodic_returns.npy'), episodic_returns)
    np.save(os.path.join(save_dir, 'sedimentary_returns.npy'), sedimentary_returns)
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

