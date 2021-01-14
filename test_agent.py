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
buffer_size = 300

agent = IntrinsicMotivationAgent(dim_latent,dim_view,dim_act,num_act)
buf = OnPolicyBuffer(dim_latent, dim_act, buffer_size)

# encoder
imgs = np.random.uniform(0,1,(100,128,128,1))
lats_m, lats_logs = agent.encoder(imgs)
lats = agent.encode(imgs)
# decoder
ls= np.random.normal(0,1,(100,8))
rec_imgs = agent.decoder(ls)
rec_imgs_ = agent.decode(ls, apply_sigmoid=True)
# imaginator
acts = tf.convert_to_tensor(np.random.randint(0,10,(100,1)), dtype=tf.float32)
ss = lats.sample()
imn_ss = agent.imaginator([ss, acts])
imn_ss_ = agent.imagine(ss, acts)
# actor
logits = agent.actor(ss)
vs = agent.critic(ss)
vals = agent.estimate_value(ss)
acts_, vals_, logp_as = agent.make_decision(ss)
# reward
img = np.random.uniform(0,1,(1,128,128,1))
distr = agent.encode(img)
state = distr.sample()
act = tf.convert_to_tensor(np.random.randint(0,10,(1,1)), dtype=tf.float32)
imgn = agent.imagine(state, act)
nimg = np.random.uniform(0,1,(1,128,128,1))
ndistr = agent.encode(nimg)
r = agent.compute_intrinsic_reward(imgn, ndistr)

# collect experience
img = np.random.uniform(0,1,(1,128,128,1))
lat = agent.encode(img)
s = lat.sample()
a, v, logp_a = agent.make_decision(s)
imn = agent.imagine(s, np.reshape(a, (1,1)).astype(np.float32))
for _ in range(buffer_size):
    nimg = np.random.uniform(0,1,(1,128,128,1))
    nlat = agent.encode(nimg)
    ns = nlat.sample()
    r = agent.compute_intrinsic_reward(imn, nlat)
    buf.store(
        np.squeeze(lat.mean()), 
        np.squeeze(lat.stddev()), 
        np.squeeze(nlat.mean()), 
        np.squeeze(nlat.stddev()), 
        np.squeeze(s), 
        np.squeeze(ns), 
        np.squeeze(imn), 
        a, 
        r, 
        v, 
        logp_a
    )
    s = ns
    a, v, logp_a = agent.make_decision(s)
    imn = agent.imagine(s, np.reshape(a, (1,1)).astype(np.float32))
_, v, _ = agent.make_decision(s)
buf.finish_path(v)

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



