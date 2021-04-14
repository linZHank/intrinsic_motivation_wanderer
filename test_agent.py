import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
tfd = tfp.distributions
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)

from agents.ima_macromphalus import IntrinsicMotivationAgent, OnPolicyBuffer

img_dir = '/media/palebluedotian0/Micron1100_2T/playground/intrinsic_motivation_wanderer/macromphalus_experience/2021-03-05-17-42/views'
img_files = os.listdir(img_dir)
agent = IntrinsicMotivationAgent()
buf = OnPolicyBuffer(max_size=100)
 
# collect experience
# i = np.random.uniform(0,1,(1,128,128,1))
img = cv2.imread(os.path.join(img_dir, '0.jpg'), 0)/255.
img.resize(1,128,128,1)
mu, ls = agent.vae.encoder(img)
o = agent.vae.reparameterize(mu, ls)
a, v, l = agent.ac.make_decision(tf.expand_dims(o,0))
mu2_, ls2_= agent.imaginator(tf.expand_dims(o,0), tf.reshape(a,(1,1)))
o2_ = agent.vae.reparameterize(mu2_, ls2_)
for i in range(100):
    img2 = cv2.imread(os.path.join(img_dir, str((i+1)*30)+'.jpg'), 0)/255.
    img2.resize(1,128,128,1)
    mu2, ls2 = agent.vae.encoder(img2)
    o2 = agent.vae.reparameterize(mu2, ls2)
    r = agent.compute_intrinsic_reward(mu2, ls2, o2_)
    logging.debug('act: {}, val: {}, lpa: {}, rew: {}'.format(a,v,l,r))
    buf.store(
        o, 
        a, 
        r, 
        v, 
        l, 
        o2,
        o2_,
        mu,
        ls,
        mu2,
        ls2,
        mu2_,
        ls2_,
    )
    o = o2
    mu = mu2
    ls = ls2
_, v, _ = agent.ac.make_decision(tf.expand_dims(o,0))
buf.finish_path(v)

# train actor critic
data = buf.get()
loss_pi, loss_val, loss_info = agent.ac.train(data, 80)
# train_vae
dataset = tf.keras.preprocessing.image_dataset_from_directory(os.path.dirname(img_dir), color_mode='grayscale', image_size=(128,128), batch_size=128)
dataset = dataset.map(lambda x, y: x/255.)
loss_elbo = agent.vae.train(dataset, num_epochs=2)

fig, ax = plt.subplots(figsize=(10,20), nrows=10, ncols=3)
for img in dataset.take(1):
    mu, logsigma = agent.vae.encoder(img)
    sample = agent.vae.reparameterize(mu, tf.math.exp(logsigma))
    rec_mean = agent.vae.decoder(mu) 
    rec_sample = agent.vae.decoder(sample) 
    for i in range(10):
        ax[i,0].imshow(img[i,:,:,0], cmap='gray')
        ax[i,0].axis('off')
        ax[i,1].imshow(rec_mean[i,:,:,0], cmap='gray')
        ax[i,1].axis('off')
        ax[i,2].imshow(rec_sample[i,:,:,0], cmap='gray')
        ax[i,2].axis('off')
plt.tight_layout()
plt.show()

# train imaginator
loss_imn = agent.imaginator.train(data, 80)
