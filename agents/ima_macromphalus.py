"""
Intrinsic reward driven agent class using image as input
"""
import tensorflow as tf
import numpy as np
import scipy.signal
import tensorflow_probability as tfp
tfd = tfp.distributions
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

################################################################
"""
On-policy Replay Buffer 
"""
def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0, x1, x2]
    output:
        [x0+discount*x1+discount^2*x2, x1+discount*x2, x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class OnPolicyBuffer: # To save memory, no image will be saved. Instead, they will be saved in hard disk.

    def __init__(self, dim_state=8, dim_act=1, max_size=1000, gamma=0.99, lam=0.97):
        # params
        self.dim_state = dim_state
        self.dim_act = dim_act
        self.max_size = max_size
        self.gamma = gamma
        self.lam = lam
        # init buffers
        self.stt_buf = np.zeros(shape=(max_size, dim_state), dtype=np.float32) # state, default dtype=tf.float32
        self.act_buf = np.zeros(shape=(max_size, dim_act), dtype=np.float32) # action
        self.rew_buf = np.zeros(shape=(max_size,), dtype=np.float32) # reward
        self.val_buf = np.zeros(shape=(max_size,), dtype=np.float32) # value
        self.ret_buf = np.zeros(shape=(max_size,), dtype=np.float32) # rewards-to-go return
        self.adv_buf = np.zeros(shape=(max_size,), dtype=np.float32) # advantage
        self.lpa_buf = np.zeros(shape=(max_size,), dtype=np.float32) # logprob(action)
        # variables
        self.ptr, self.path_start_idx = 0, 0

    def store(self, stt, act, rew, val, lpa):
        assert self.ptr <= self.max_size     # buffer has to have room so you can store
        self.stt_buf[self.ptr] = stt
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.lpa_buf[self.ptr] = lpa
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma*self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr <= self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        self.stt_buf = self.stt_buf[:self.ptr] 
        self.act_buf = self.act_buf[:self.ptr] 
        self.rew_buf = self.rew_buf[:self.ptr] 
        self.val_buf = self.val_buf[:self.ptr] 
        self.ret_buf = self.ret_buf[:self.ptr] 
        self.adv_buf = self.adv_buf[:self.ptr] 
        self.lpa_buf = self.lpa_buf[:self.ptr]
        # the next three lines implement the advantage normalization trick
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(
            stt=self.stt_buf, 
            act=self.act_buf, 
            ret=self.ret_buf, 
            adv=self.adv_buf, 
            lpa=self.lpa_buf
        )
        return {k: tf.convert_to_tensor(v, dtype=tf.float32) for k,v in data.items()}
################################################################

class VAE(tf.keras.Model):

    def __init__(self, dim_view, dim_latent, **kwargs):
        super(VAE, self).__init__(name='vae', **kwargs)
        self.dim_view = dim_view
        self.dim_latent = dim_latent
        # construct encoder
        inputs_encoder = tf.keras.Input(shape=dim_view, name='encoder_inputs')
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation='relu')(inputs_encoder)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        mu = tf.keras.layers.Dense(dim_latent, name='encoder_outputs_mean')(x)
        logsigma = tf.keras.layers.Dense(dim_latent, name='encoder_outputs_logstddev')(x)
        self.encoder = tf.keras.Model(inputs=inputs_encoder, outputs = [mu, logsigma])
        # construct decoder
        inputs_decoder = tf.keras.Input(shape=(dim_latent,), name='decoder_inputs')
        x = tf.keras.layers.Dense(units=16*16*64, activation='relu')(inputs_decoder) 
        x = tf.keras.layers.Reshape(target_shape=(16, 16, 64))(x)
        x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', activation='relu')(x)
        outputs_decoder = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=4, strides=1, padding='same', name='decoder_outputs')(x)
        self.decoder = tf.keras.Model(inputs=inputs_decoder, outputs=outputs_decoder)

    @tf.function
    def reparameterize(self, mean, std):
        eps = tf.random.normal(shape=mean.shape)
        return mean + eps*std 

    @tf.function
    def encode(self, img):
        mean, logstd = self.encoder(img)
        return mean, logstd

    @tf.function
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.math.sigmoid(logits)
            return probs
        return logits


class CategoricalActor(tf.keras.Model):

    def __init__(self, dim_obs, num_act, **kwargs):
        super(CategoricalActor, self).__init__(name='categorical_actor', **kwargs)
        self.dim_obs=dim_obs
        self.num_act=num_act
        self.policy_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=dim_obs, name='actor_inputs'),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(num_act, activation=None, name='actor_outputs')
            ]
        )

    def _distribution(self, obs):
        logits = tf.squeeze(self.policy_net(obs)) # squeeze to deal with size 1
        d = tfd.Categorical(logits=logits)
        return d

    def _logprob(self, distribution, act):
        logp_a = distribution.log_prob(act) # get log probability from a tfp distribution
        return logp_a
        
    def call(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._logprob(pi, act)
        return pi, logp_a

class Critic(tf.keras.Model):
    def __init__(self, dim_obs, **kwargs):
        super(Critic, self).__init__(name='critic', **kwargs)
        self.value_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=dim_obs, name='critic_inputs'),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(1, activation=None, name='critic_outputs')
            ]
        )

    @tf.function
    def call(self, obs):
        return tf.squeeze(self.value_net(obs))

class IntrinsicMotivationAgent(tf.keras.Model):
    def __init__(self, dim_view=(128,128,1), dim_latent=8, num_act=10, clip_ratio=0.2, beta=0., target_kl=0.1, lr_vae=1e-4, lr_actor=3e-4, lr_critic=1e-3, **kwargs):
        super(IntrinsicMotivationAgent, self).__init__(name='ima', **kwargs)
        # parameters
        self.dim_latent = dim_latent
        self.dim_view = dim_view
        self.dim_act = dim_act
        self.clip_ratio = clip_ratio
        self.beta = beta
        self.target_kl = target_kl
        # modules
        self.vae = VAE(dim_view, dim_latent)
        self.actor = CategoricalActor(dim_latent, num_act)
        self.critic = Critic(dim_latent)
        # optimizers
        self.optimizer_vae = tf.keras.optimizers.Adam(learning_rate=lr_vae)
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=lr_critic)

    @tf.function
    def make_decision(self, obs):
        pi = self.actor._distribution(obs)
        act = pi.sample()
        logp_a = self.actor._logprob(pi, act)
        val = self.critic(obs)

        return act, val, logp_a

    def train_vae(self, dataset, num_epochs):

        def log_normal_pdf(sample, mean, logstd, raxis=1):
            log_pi = tf.math.log(2.*np.pi)
            return tf.math.reduce_sum(-.5*(((sample-mean)*tf.math.exp(-logstd))**2 + 2*logstd + log_pi), axis=-1)
        elbo_per_epoch = []
        for e in range(num_epochs):
            step_cntr = 0
            metrics_mean = tf.keras.metrics.Mean()
            for batch in dataset:
                with tf.GradientTape() as tape:
                    tape.watch(self.encoder.trainable_weights + self.decoder.trainable_weights)
                    mean, logstd = self.encode(batch)
                    z = self.reparameterize(mean, logstd)
                    reconstructed = self.decode(z)
                    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=reconstructed, labels=batch)
                    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
                    logpz = log_normal_pdf(z, 0., 0.)
                    logqz_x = log_normal_pdf(z, mean, logstd)
                    loss_vae = -tf.math.reduce_mean(logpx_z + logpz - logqz_x)
                    loss_vae_mean = metrics_mean(loss_vae)
                gradients_vae = tape.gradient(loss_vae, self.encoder.trainable_weights+self.decoder.trainable_weights)
                self.optimizer_vae.apply_gradients(zip(gradients_vae, self.encoder.trainable_weights+self.decoder.trainable_weights))
                step_cntr += 1
                logging.debug("Epoch: {}, Step: {} \nVAE Loss: {}".format(e+1, step_cntr, loss_vae))
            elbo = -loss_vae_mean
            logging.info("Epoch {} ELBO: {}".format(e+1,elbo))
            elbo_per_epoch.append(elbo)
        return elbo_per_epoch

    def train_ppo(self, data, num_epochs):
        # Update actor
        ep_loss_pi = []
        ep_kld = []
        ep_ent = []
        for e in range(num_epochs):
            logging.debug("Staring actor training epoch: {}".format(e+1))
            with tf.GradientTape() as tape:
                tape.watch(self.actor.trainable_variables)
                pi, lpa = self.actor(data['obs'], data['act'])
                ratio = tf.math.exp(lpa - data['lpa']) # pi/old_pi
                clip_adv = tf.math.multiply(tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio), data['adv'])
                approx_kld = data['lpa'] - lpa
                ent = tf.math.reduce_sum(pi.entropy(), axis=-1)
                obj = tf.math.minimum(ratio*data['adv'], clip_adv) + self.beta*ent
                loss_pi = -tf.math.reduce_mean(obj)
            # gradient descent actor weights
            grads_actor = tape.gradient(loss_pi, self.actor.trainable_variables)
            self.optimizer_actor.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
            ep_loss_pi.append(loss_pi)
            ep_kld.append(tf.math.reduce_mean(approx_kld))
            ep_ent.append(tf.math.reduce_mean(ent))
            # log epoch
            logging.info("\n----Actor Training----\nEpoch :{} \nLoss: {} \nKLDivergence: {} \nEntropy: {}".format(
                e+1,
                loss_pi,
                ep_kld[-1],
                ep_ent[-1],
            ))
            # early cutoff due to large kl-divergence
            if ep_kld[-1] > self.target_kld:
                logging.warning("\nEarly stopping at epoch {} due to reaching max kl-divergence.\n".format(e+1))
                break
        mean_loss_pi = tf.math.reduce_mean(ep_loss_pi)
        mean_ent = tf.math.reduce_mean(ep_ent)
        mean_kld = tf.math.reduce_mean(ep_kld)
        # Update critic
        ep_loss_val = []
        for e in range(num_epochs):
            logging.debug("Starting critic training epoch: {}".format(e+1))
            with tf.GradientTape() as tape:
                tape.watch(self.critic.trainable_variables)
                # loss_val = self.mse(data['ret'], self.critic(data['obs']))
                loss_val = tf.keras.losses.MSE(data['ret'], self.critic(data['obs']))
            # gradient descent critic weights
            grads_critic = tape.gradient(loss_val, self.critic.trainable_variables)
            self.optimizer_critic.apply_gradients(zip(grads_critic, self.critic.trainable_variables))
            ep_loss_val.append(loss_val)
            # log loss_v
            logging.info("\n----Critic Training----\nEpoch :{} \nLoss: {}".format(
                e+1,
                loss_val
            ))
        mean_loss_val = tf.math.reduce_mean(ep_loss_val)

        return mean_loss_pi, mean_loss_val, dict(kld=mean_kld, entropy=mean_ent)

    


data_dir = '/media/palebluedotian0/Micron1100_2T/playground/intrinsic_motivation_wanderer/experience/2021-01-20-17-07/'
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    color_mode='grayscale',
    image_size=(128, 128),
    batch_size=64
)
dataset = dataset.map(lambda x, y: x/254.)
vae = VAE()
loss_elbo = vae.train(dataset, num_epochs=20)
import os
import matplotlib.pyplot as plt
import cv2
sf = np.load(os.path.join(data_dir, 'stepwise_frames.npy'))
frames = sf[-11:]-1
fig, ax = plt.subplots(nrows=10, ncols=2)
for i in range(len(frames)-1):
    # original 
    ori = cv2.imread(os.path.join(data_dir, 'views', str(frames[i])+'.jpg'), 0)/254.
    ax[i,0].imshow(ori, cmap='gray')
    ax[i,0].axis('off')
    # reconstruction
    mu, logsigma = vae.encode(np.expand_dims(ori,0))
    z = vae.reparameterize(mu, tf.math.exp(logsigma))
    rec = vae.decode(z)
    ax[i,1].imshow(rec[0,:,:,0], cmap='gray')
    ax[i,1].axis('off')

plt.subplots_adjust(wspace=0, hspace=.1)
plt.tight_layout()
plt.show()


