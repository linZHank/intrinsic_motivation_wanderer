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

from agents.ima_macromphalus import IntrinsicMotivationAgent

import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

# Parameters
dim_view = (128,128,1)
dim_latent = 8
num_act = 10
dim_act = 1
version = 'macromphalus'

# Load agent
brain = IntrinsicMotivationAgent(dim_view, dim_latent, num_act, dim_act)
model_dir = os.path.join(sys.path[0], 'model_dir', version, '2021-03-09-17-29')
brain.vae.encoder.encoder_net = tf.keras.models.load_model(os.path.join(model_dir, 'encoder'))
brain.vae.decoder.decoder_net = tf.keras.models.load_model(os.path.join(model_dir, 'decoder'))
brain.imaginator.dynamics_net = tf.keras.models.load_model(os.path.join(model_dir, 'imaginator'))
brain.ac.actor.policy_net = tf.keras.models.load_model(os.path.join(model_dir, 'actor'))
brain.ac.critic.value_net = tf.keras.models.load_model(os.path.join(model_dir, 'critic'))

# Load data
data_dir = '/media/palebluedotian0/Micron1100_2T/playground/intrinsic_motivation_wanderer/macromphalus_experience/2021-03-09-17-29/'
dataset_views = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    color_mode='grayscale',
    image_size=dim_view[:2],
    batch_size=64
)
dataset_views = dataset_views.map(lambda x, y: x/255.)

# train vae
loss_elbo = brain.vae.train(dataset=dataset_views, num_epochs=20)

# train imaginator
actions = np.load(os.path.join(data_dir, 'action_data.npy'))
frames = np.load(os.path.join(data_dir, 'stepwise_frames.npy'))
views = np.zeros([frames.shape[0]+1]+list(dim_view))
views[0] = np.expand_dims(cv2.imread(os.path.join(data_dir, 'views/0.jpg'), 0), -1)
for i, f in enumerate(frames):
    views[i+1] = np.expand_dims(cv2.imread(os.path.join(data_dir, 'views', str(f)+'.jpg'), 0), -1)
obs, _ = brain.vae.encoder(views[:-1])
mu, logsigma = brain.vae.encoder(views)
data = dict(
    obs=mu[:-1],
    act=tf.convert_to_tensor(actions),
    nobs=mu[1:] 
)
loss_dyna = brain.imaginator.train(data=data, num_epochs=100)
