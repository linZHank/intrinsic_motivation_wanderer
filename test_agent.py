import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

from agents.intrinsic_motivation_agent import IntrinsicMotivationAgent, OnPolicyBuffer

dim_latent = 8
dim_view = (128,128,1)
dim_act = 1
num_act = 10

agent = IntrinsicMotivationAgent(dim_latent,dim_view,dim_act,num_act)

# encoder
imgs = np.random.uniform(0,1,(100,128,128,1))
enc_m, enc_logs = agent.encoder(imgs)
enc_distr = agent.encode(imgs)
# decoder
l= np.random.normal(0,1,(100,8))
rec_imgs = agent.decoder(l)
rec_imgs_ = agent.decode(l, apply_sigmoid=True)
# imaginator
acts = tf.convert_to_tensor(np.random.randint(0,10,(100,1)), dtype=tf.float32)
s = enc_distr.sample()
imn_s = agent.imaginator([s, acts])
imn_s_ = agent.imagine(s, acts)
# actor
logits = agent.actor(s)
v= agent.critic(s)
val = agent.estimate_value(s)
acts_, val_, logp_a = agent.make_decision(s)
# reward
img = np.random.uniform(0,1,(1,128,128,1))
distr = agent.encode(img)
ls = distr.sample()
act = tf.convert_to_tensor(np.random.randint(0,10,(1,1)), dtype=tf.float32)
imgn = agent.imagine(ls, act)
nimg = np.random.uniform(0,1,(1,128,128,1))
ndistr = agent.encode(nimg)
r = agent.compute_intrinsic_reward(imgn, ndistr)

# # train vae
# bsize = 32
# imgs_tensor = tf.convert_to_tensor(imgs, dtype=tf.float32)
# dataset = tf.data.Dataset.from_tensor_slices(imgs_tensor).batch(bsize)
# agent.train_autoencoder(dataset, num_epochs=10)
# # train actor critic
# rews = np.random.normal(0,2,100)
# buf = OnPolicyBuffer(dim_latent=dim_latent, dim_act=dim_act, size=100)
# for i in range(100):
#     buf.store(np.squeeze(enc_distr.mean()[i]), np.squeeze(enc_distr.stddev()[i]), np.squeeze(imn_distr.mean()[i]), np.squeeze(imn_distr.stddev()[i]), acts_[i], rews[i], val_[i], logp_a[i])
# buf.finish_path(0)    
# data = buf.get()
# loss_pi, loss_v, info = agent.train_policy(data, 10)
# # train imaginator
# agent.train_imaginator(data, 10)



