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

dim_latent = 16
dim_view = (128,128,1)
dim_act = 1
num_act = 10
# Parameters
data_dir = '/media/palebluedotian0/Micron1100_2T/playground/intrinsic_motivation_wanderer/experience/2020-12-15-18-10/'
# Instantiate agent
brain = IntrinsicMotivationAgent(dim_latent,dim_view,dim_act,num_act)

# Create image dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    color_mode='grayscale',
    validation_split=0.1,
    subset="training",
    seed=123,
    image_size=(height, width),
    batch_size=batch_size
)
dataset = dataset.map(lambda x, y: x/255.)
# Load replay buffer
replay_data = np.load(data_dir+'replay_data.npy', allow_pickle=True).item()

# train autoencoder
num_iters = 100
batch_size = 32
brain.train_autoencoder(dataset, num_epochs=num_iters)

# train imaginator
agent.train_imaginator(data, num_iters=num_iters)

# Train policy
loss_pi, loss_v, loss_info = brain.train_policy(replay_data, num_iters=num_iters)


