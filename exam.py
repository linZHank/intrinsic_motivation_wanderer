import sys
import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from agents.intrinsic_motivation_agent import IntrinsicMotivationAgent 


# Parameters
dim_latent = 16
dim_view = (128,128,1)
dim_act = 1
num_act = 10
seed = 'belauensis'

# Load agent
brain = IntrinsicMotivationAgent(dim_latent=dim_latent, dim_view=dim_view, dim_act=dim_act, num_act=num_act)
load_dir = os.path.join(sys.path[0], 'model_dir', seed, '2021-01-18-12-59')
brain.encoder = tf.keras.models.load_model(os.path.join(load_dir, 'encoder'))
brain.decoder = tf.keras.models.load_model(os.path.join(load_dir, 'decoder'))
brain.imaginator = tf.keras.models.load_model(os.path.join(load_dir, 'imaginator'))
brain.actor = tf.keras.models.load_model(os.path.join(load_dir, 'actor'))
brain.critic = tf.keras.models.load_model(os.path.join(load_dir, 'critic'))

# Load data
data_dir = '/media/palebluedotian0/Micron1100_2T/playground/intrinsic_motivation_wanderer/experience/2021-01-18-13-18/'
data = np.load(data_dir+'replay_data.npy', allow_pickle=True).item()
sf = np.load(os.path.join(data_dir, 'stepwise_frames.npy'))
frames = sf[-11:]-1

# Plot frames
fig, ax = plt.subplots(figsize=(10,20), nrows=10, ncols=5)
for i in range(len(frames)-1):
    # original present
    pre = cv2.imread(os.path.join(data_dir, 'views', str(frames[i])+'.jpg'), 0)
    ax[i,0].imshow(pre, cmap='gray')
    ax[i,0].axis('off')
    # reconstructed present
    rec_p = brain.decode(np.expand_dims(data['state'][-10+i],0))
    ax[i,1].imshow(rec_p[0,:,:,0], cmap='gray')
    ax[i,1].axis('off')
    # imagination
    imn = brain.decode(np.expand_dims(data['imn'][-10+i],0))
    ax[i,2].imshow(imn[0,:,:,0], cmap='gray')
    ax[i,2].axis('off')
    # reconstructed future
    rec_f = brain.decode(np.expand_dims(data['nstate'][-10+i],0))
    ax[i,3].imshow(rec_f[0,:,:,0], cmap='gray')
    ax[i,3].axis('off')
    # original future
    fut = cv2.imread(os.path.join(data_dir, 'views', str(frames[i+1])+'.jpg'), 0)
    ax[i,4].imshow(fut, cmap='gray')
    ax[i,4].axis('off')

plt.subplots_adjust(wspace=0, hspace=.1)
plt.tight_layout()
plt.savefig(os.path.join(load_dir, 'exam.png'))
plt.show()
