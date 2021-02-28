import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
tfd = tfp.distributions
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)

from agents.ima_macromphalus import IntrinsicMotivationAgent, OnPolicyBuffer

agent = IntrinsicMotivationAgent()
buf = OnPolicyBuffer(max_size=100)
 
# collect experience
i = np.random.uniform(0,1,(1,128,128,1))
o, _ = agent.vae.encoder(i)
a, v, l = agent.ac.make_decision(tf.expand_dims(o,0))
mu, logsigma = agent.imaginator(tf.expand_dims(o,0), tf.reshape(a,(1,1)))
for _ in range(100):
    mu_imn, ls_imn = agent.imaginator(tf.expand_dims(o,0), tf.reshape(a,(1,1)))
    i2 = np.random.uniform(0,1,(1,128,128,1))
    mu, ls = agent.vae.encoder(i2)
    r = agent.compute_intrinsic_reward(mu_imn, ls_imn, mu)
    buf.store(
        o, 
        a, 
        r, 
        v, 
        l, 
        mu,
        ls
    )
    o = mu
    a, v, l = agent.ac.make_decision(tf.expand_dims(o,0))
_, v, _ = agent.ac.make_decision(tf.expand_dims(o,0))
buf.finish_path(v)

# train actor critic
data = buf.get()
loss_pi, loss_val, loss_info = agent.ac.train(data, 80)
# train_vae
# data_dir = '/media/palebluedotian0/Micron1100_2T/playground/intrinsic_motivation_wanderer/experience/2021-01-20-17-07'
# dataset = tf.keras.preprocessing.image_dataset_from_directory(data_dir, color_mode='grayscale', image_size=(128,128), batch_size=32)
# dataset = dataset.map(lambda x, y: x/255.)
# loss_elbo = agent.vae.train(dataset, num_epochs=20)
# 
# fig, ax = plt.subplots(figsize=(10,20), nrows=10, ncols=2)
# for imgs in dataset.take(1):
#     z, _ = agent.vae.encoder(imgs)
#     recs = agent.vae.decoder(z) 
#     for i in range(10):
#         ax[i,0].imshow(imgs[i,:,:,0], cmap='gray')
#         ax[i,0].axis('off')
#         ax[i,1].imshow(recs[i,:,:,0], cmap='gray')
#         ax[i,1].axis('off')
# plt.show()

# train imaginator
loss_imn = agent.imaginator.train(data, 80)
