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
ret, frame = eye.read()
obs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Get agent ready
total_steps = 30
max_ep_len = 10
dim_latent = 8
dim_obs=(128,128,1)
dim_act=1
num_act = 10
brain = IntrinsicMotivationAgent(act_type='discrete', dim_latent=dim_latent, dim_obs=dim_obs, dim_act=dim_act, num_act=num_act)
brain.imagine_state(np.expand_dims(np.expand_dims(obs, -1), 0)) # (128,128) -> (1,128,128,1)
replay_buffer = OnPolicyBuffer(dim_act=dim_act, size=num_steps, gamma=.99, lam=.97)
# Set experience saving dir
save_dir = '/ssd/mecanum_experience/' + datetime.now().strftime("%Y-%m-%d-%H-%M") + '/'
if not os.path.exists(save_dir):
    try:
        os.makedirs(save_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
# Preapare for experience collecting
frame_counter = 0
step_counter = 0
buffer_actions = []
buffer_frame_counters = []
time_elapse = 0
prev_time_elapse = -1
start_time = time.time()
# Main loop
try:
    for st in range(total_steps):
        ret, frame = eye.read()
        obs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(save_dir, str(frame_counter)+'.jpg'), obs)
        if not int(time_elapse)%2 and int(prev_time_elapse)%2: # change mecanum's behavior every 2 sec
            act, val, logp = brain.pi_of_a_given_s(np.expand_dims(np.expand_dims(obs, -1), 0)) 
            wanderer.set_action(int(act))


            # encoded_state = brain.image_encoder.encode(np.expand_dims(np.expand_dims(obs, -1), 0))
            rew = brain.compute_intrinsic_reward(np.expand_dims(np.expand_dims(obs, -1), 0))


#     while step_counter < num_steps: 
#         ret, frame = cap.read()
#         obs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         cv2.imwrite(os.path.join(save_dir, str(frame_counter)+'.jpg'), obs)
#         # Display the resulting frame, comment next 3 lines out if headless
#         # cv2.imshow('frame', obs)
#         # if cv2.waitKey(1) & 0xFF == ord('q'):
#         #     break
#         if not int(time_elapse)%2 and int(prev_time_elapse)%2: # change mecanum's behavior every 2 sec
#             act, val, logp = agent.pi_of_a_given_s(np.expand_dims(np.expand_dims(obs, -1), 0))
#             encoded_state = agent.autoencoder.encode(np.expand_dims(np.expand_dims(obs, -1), 0))
#             imagined_state = agent.imagine(np.expand_dims(np.expand_dims(obs, -1), 0))
#             rew = np.linalg.norm(encoded_state[0].numpy()-imagined_state[0].numpy())
#             print("\nstep: {} \nencoded state: {} \nimagined state: {}\n".format(step_counter+1, encoded_state, imagined_state))
#             mec.set_action(int(act))
#             step_counter += 1
#             replay_buffer.store(act=act, rew=rew, val=val, logp=logp)
#             buffer_frame_counters.append(frame_counter)
# 
#         prev_time_elapse = time_elapse
#         time_elapse = time.time() - start_time
#         # print("prev te: {}, te: {}".format(prev_time_elapse, time_elapse)) # debug
#         frame_counter+=1
#     replay_data = replay_buffer.get()
#     np.save(os.path.join(save_dir, 'replay_buffer.npy'), replay_data)
#     np.save(os.path.join(save_dir, 'buffer_frame_counters.npy'), buffer_frame_counters)

except KeyboardInterrupt:    
    print("\r\nctrl + c:")
    cap.release()
    cv2.destroyAllWindows()
    mec.halt()
    exit()

# When everything done, release the capture and stop motors
cap.release()
cv2.destroyAllWindows()
mec.halt()

