#!/usr/bin/python3
"""
Use this script on the computing machine. The robot learns through training.
"""
import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf

from agents.intrinsic_motivation_agent import IntrinsicMotivationAgent

import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

# Parameters
dim_latent = 16
dim_view = (128,128,1)
dim_act = 1
num_act = 10
seed = 'belauensis'
num_epochs = 100
batch_size = 32

# Load agent
brain = IntrinsicMotivationAgent(dim_latent=dim_latent, dim_view=dim_view, dim_act=dim_act, num_act=num_act)
load_dir = os.path.join(sys.path[0], 'model_dir', seed, '2021-01-19-18-05')
brain.encoder = tf.keras.models.load_model(os.path.join(load_dir, 'encoder'))
brain.decoder = tf.keras.models.load_model(os.path.join(load_dir, 'decoder'))
brain.imaginator = tf.keras.models.load_model(os.path.join(load_dir, 'imaginator'))
brain.actor = tf.keras.models.load_model(os.path.join(load_dir, 'actor'))
brain.critic = tf.keras.models.load_model(os.path.join(load_dir, 'critic'))

# Load data
data_dir = '/media/palebluedotian0/Micron1100_2T/playground/intrinsic_motivation_wanderer/experience/2021-01-20-14-54/'
dataset_views = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    color_mode='grayscale',
    image_size=(128, 128),
    batch_size=batch_size
)
dataset_views = dataset_views.map(lambda x, y: x/255.)
replay_data = np.load(data_dir+'replay_data.npy', allow_pickle=True).item()

# train autoencoder
loss_elbo = brain.train_autoencoder(dataset_views, num_epochs=int(num_epochs/5))

# train imaginator
loss_i = brain.train_imaginator(replay_data, num_epochs=num_epochs)

# Train policy
loss_pi, loss_v, loss_info = brain.train_policy(replay_data, num_epochs=num_epochs)

# Save models
save_dir = os.path.join(sys.path[0], 'model_dir', seed, data_dir.split('/')[-2]) 
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
brain.encoder.save(os.path.join(save_dir, 'encoder'))
brain.decoder.save(os.path.join(save_dir, 'decoder'))
brain.imaginator.save(os.path.join(save_dir, 'imaginator'))
brain.actor.save(os.path.join(save_dir, 'actor'))
brain.critic.save(os.path.join(save_dir, 'critic'))

# Plot episode return
prevdata_dir = '/media/palebluedotian0/Micron1100_2T/playground/intrinsic_motivation_wanderer/experience/2021-01-19-18-05/'
if not prevdata_dir==data_dir:
    prev_eprets = np.load(os.path.join(prevdata_dir, 'combined_episodic_returns.npy'))
    eprets = np.load(os.path.join(data_dir, 'episodic_returns.npy'))
    combined_episodic_returns = np.concatenate((prev_eprets, eprets))
    np.save(os.path.join(data_dir, 'combined_episodic_returns.npy'), combined_episodic_returns)
    plt.plot(combined_episodic_returns)
    plt.show()



