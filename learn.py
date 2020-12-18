#!/usr/bin/python3
"""
Use this script on the computing machine. The robot learns through training.
"""
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf

from agents.intrinsic_motivation_agent import OnPolicyBuffer, IntrinsicMotivationAgent

import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)
# Parameters
height = 128
width = 128
dim_latent = 8
dim_origin=(128,128,1)
dim_obs=(128,128,2)
dim_act = 1
num_act = 10
data_dir = '/media/palebluedotian0/Micron1100_2T/playground/intrinsic_motivation_wanderer/experience/2020-12-15-18-10/'
# Instantiate agent
brain = IntrinsicMotivationAgent(dim_latent=dim_latent, dim_origin=dim_origin, act_type='discrete', dim_obs=dim_obs, dim_act=dim_act, num_act=num_act)
# Load replay buffer
replay_data = np.load(data_dir+'replay_data.npy', allow_pickle=True).item()
# Train policy
num_iters = 100
loss_pi, loss_v, loss_info = brain.train_policy(replay_data, num_iters)

# Load images
# batch_size = 128
# image_dataset = tf.keras.preprocessing.image_dataset_from_directory(
#     directory=data_dir,
#     color_mode='grayscale',
#     image_size=(height, width),
#     batch_size=batch_size,
# )
# image_dataset = image_dataset.map(lambda x, y: x/255.)
# # Train VAE
# num_epochs = 0
# elbo_per_epoch = brain.train_autoencoder(image_dataset, num_epochs)

