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
# num_epochs = 100
# batch_size = 64

# Load agent
load_dir = os.path.join(sys.path[0], 'model_dir', version, '2021-04-14-15-39')
brain = IntrinsicMotivationAgent(dim_view=dim_view, dim_latent=dim_latent, num_act=num_act, dim_act=dim_act)
brain.vae.encoder.encoder_net = tf.keras.models.load_model(os.path.join(load_dir, 'encoder'))
brain.vae.decoder.decoder_net = tf.keras.models.load_model(os.path.join(load_dir, 'decoder'))
brain.imaginator.dynamics_net = tf.keras.models.load_model(os.path.join(load_dir, 'imaginator'))
brain.ac.actor.policy_net = tf.keras.models.load_model(os.path.join(load_dir, 'actor'))
brain.ac.critic.value_net = tf.keras.models.load_model(os.path.join(load_dir, 'critic'))

# Load data
data_dir = '/media/palebluedotian0/Micron1100_2T/playground/intrinsic_motivation_wanderer/macromphalus_experience/2021-04-22-10-58'
dataset_views = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    color_mode='grayscale',
    image_size=dim_view[:2],
    batch_size=64
)
dataset_views = dataset_views.map(lambda x, y: x/255.)
replay_data = np.load(os.path.join(data_dir, 'replay_data.npy'), allow_pickle=True).item()

# Train policy
train_ac = input('Train actor-critic? [y/n]')
if train_ac=='y':
    loss_pi, loss_val, loss_info = brain.ac.train(replay_data, num_epochs=50)
else:
    loss_pi, loss_val = 0, 0
# train autoencoder
loss_vae = brain.vae.train(dataset_views, num_epochs=20)
# train imaginator
loss_imn = brain.imaginator.train(replay_data, num_epochs=50)


# Save models
save_dir = os.path.join(sys.path[0], 'model_dir', version, datetime.now().strftime("%Y-%m-%d-%H-%M")) 
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
brain.vae.encoder.encoder_net.save(os.path.join(save_dir, 'encoder'))
brain.vae.decoder.decoder_net.save(os.path.join(save_dir, 'decoder'))
brain.imaginator.dynamics_net.save(os.path.join(save_dir, 'imaginator'))
brain.ac.actor.policy_net.save(os.path.join(save_dir, 'actor'))
brain.ac.critic.value_net.save(os.path.join(save_dir, 'critic'))
# save losses
np.save(os.path.join(save_dir, 'loss_pi.npy'), loss_pi)
np.save(os.path.join(save_dir, 'loss_val.npy'), loss_val)
np.save(os.path.join(save_dir, 'loss_vae.npy'), loss_vae)
np.save(os.path.join(save_dir, 'loss_imn.npy'), loss_imn)
# save sample figures
fig, ax = plt.subplots(figsize=(10,20), nrows=10, ncols=3)
for img in dataset_views.take(1):
    mu, logsigma = brain.vae.encoder(img)
    sample = brain.vae.reparameterize(mu, logsigma)
    rec_mean = brain.vae.decoder(mu) 
    rec_sample = brain.vae.decoder(sample) 
    for i in range(10):
        ax[i,0].imshow(img[i,:,:,0], cmap='gray')
        ax[i,0].axis('off')
        ax[i,1].imshow(rec_mean[i,:,:,0], cmap='gray')
        ax[i,1].axis('off')
        ax[i,2].imshow(rec_sample[i,:,:,0], cmap='gray')
        ax[i,2].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'samples.png'))

# Plot episode return
data_dir0 = '/media/palebluedotian0/Micron1100_2T/playground/intrinsic_motivation_wanderer/macromphalus_experience/2021-04-14-15-22'
if not data_dir0==data_dir:
    acc_rets = np.load(os.path.join(data_dir0, 'accumulated_returns.npy'))
    ret = np.load(os.path.join(data_dir, 'return.npy')) # current return
    acc_rets = np.append(acc_rets, ret)
    np.save(os.path.join(data_dir, 'accumulated_returns.npy'), acc_rets)
    plt.clf()
    plt.plot(acc_rets)
    plt.show()



