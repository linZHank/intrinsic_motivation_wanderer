#!/usr/bin/python3
"""
Deploy this script on the robot. The robot will play and collect experience data according to its imagination.
"""
import os
import time
import numpy as np
import cv2
from datetime import datetime
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

from drivers.mecanum_driver import MecanumDriver
from agents.intrinsic_motivation_agent import OnPolicyBuffer, IntrinsicMotivationAgent

# Parameters
total_steps = 300
max_ep_len = 10
dim_latent = 8
dim_view = (128,128,1)
dim_state = 4*dim_latent # (mean_encode, logstd_encode, mean_imagine, logstd_imagine)
dim_act = 1
num_act = 10
# Get mecanum driver ready
wheels = MecanumDriver() # need integrate mecdriver into agent in next version
# Get camera ready
# eye = cv2.VideoCapture(0) # usb webcam for debugging
eye = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)128, height=(int)128, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)
eye.get(cv2.CAP_PROP_FPS)
# Get agent ready
brain = IntrinsicMotivationAgent(dim_latent=dim_latent, dim_view=dim_view, act_type='discrete', dim_state=dim_state, dim_act=dim_act, num_act=num_act)
memory = OnPolicyBuffer(dim_state=dim_state, dim_latent=dim_latent, dim_act=dim_act, size=total_steps, gamma=.99, lam=.97)
# Generate first imagination and action
ret, frame = eye.read() # obs = env.reset()
view = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255. # from 0~255 to 0~1
view.resize(1,128,128,1)
latent = brain.encode(view) 
act, val, logp = brain.make_decision(latent) 
wheels.set_action(int(act))
imagination = brain.imagine(latent, act)
logging.info("\n====Ignition====\n")
# Preapare for experience collecting
save_dir = '/ssd/mecanum_experience/' + datetime.now().strftime("%Y-%m-%d-%H-%M") + '/'
visions_dir = save_dir+'visions'
if not os.path.exists(visions_dir):
    try:
        os.makedirs(visions_dir)
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
        cv2.imwrite(os.path.join(save_dir, 'visions', str(frame_counter)+'.jpg'), frame)
        prev_time_elapse = time_elapse
        time_elapse = time.time() - start_time
        frame_counter+=1
        if int(time_elapse)-int(prev_time_elapse): # take action every 1 sec
            next_view = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255.
            next_view.resize(1,128,128,1)
            next_latent = brain.encode(next_view)
            rew = brain.compute_intrinsic_reward(latent, imagination, next_latent)
            ep_ret+=rew
            ep_len+=1
            memory.store(np.squeeze(latent.mean()), np.squeeze(latent.stddev()), np.squeeze(imagination.mean()), np.squeeze(imagination.stddev()), act, rew, val, logp)
            step_counter+=1
            stepwise_frames.append(frame_counter)
            logging.info("\nstep: {} \nencoded view: {} \nimagination: {} \naction: {} \nvalue: {} \nlog prob: {} \nreward: {} \nepisode return: {} \nepisode length: {}".format(step_counter, (latent.mean(),latent.stddev), (imagination.mean(), imagination.stddev()), act, val, logp, rew, ep_ret, ep_len))
            # handle episode terminal
            if not step_counter%max_ep_len:
                _, val, _ = brain.make_decision(latent)
                memory.finish_path(np.squeeze(val))
                episode_counter+=1
                episodic_returns.append(ep_ret)
                sedimentary_returns.append(sum(episodic_returns)/episode_counter)
                logging.info("\n----\nTotalFrames: {} \nEpisode: {}, EpReturn: {}, EpLength: {} \n----\n".format(frame_counter, episode_counter, ep_ret, ep_len))
            # compute next obs, act, val, logp
            latent = next_latent # SUPER CRITICAL!!!
            act, val, logp = brain.make_decision(latent) 
            wheels.set_action(int(act))
            imagination = brain.imagine(latent, act)
            
    # Save valuable items
    with open(os.path.join(save_dir, 'elapsed_time.txt'), 'w') as f:
        f.write("{}".format(time.time()-start_time))
    replay_data = memory.get()
    np.save(os.path.join(save_dir, 'replay_data.npy'), replay_data)
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

