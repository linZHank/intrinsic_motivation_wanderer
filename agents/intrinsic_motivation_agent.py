"""
Intrinsic reward driven agent class using image as input
"""
import tensorflow as tf
import numpy as np
import scipy.signal
import tensorflow_probability as tfp
tfd = tfp.distributions
import logging

################################################################
"""
On-policy Replay Buffer 
"""
def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class OnPolicyBuffer: # To save memory, no image will be saved. Instead, they will be saved in hard disk.

    def __init__(self, dim_obs, dim_latent, dim_act, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros([size]+list(dim_obs), dtype=np.float32)
        self.imagination_sample_buf = np.zeros([size, dim_latent], dtype=np.float32)
        self.imagined_mean_buf = np.zeros([size, dim_latent], dtype=np.float32)
        self.imagined_stddev_buf = np.zeros([size, dim_latent], dtype=np.float32)
        self.act_buf = np.zeros((size, dim_act), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, imagination_sample, imagined_mean, imagined_stddev, act, rew, val, logp):
        assert self.ptr <= self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.imagination_sample_buf[self.ptr] = imagination_sample
        self.imagined_mean_buf[self.ptr] = imagined_mean
        self.imagined_stddev_buf[self.ptr] = imagined_stddev
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr
        # self.ptr, self.path_start_idx = 0, 0

    def get(self):
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, imagination_sample=self.imagination_sample_buf, imagined_mean=self.imagined_mean_buf, imagined_stddev=self.imagined_stddev_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: tf.convert_to_tensor(v, dtype=tf.float32) for k,v in data.items()}
################################################################

class CategoricalActor(tf.keras.Model):

    def __init__(self, dim_obs, num_act, **kwargs):
        super(CategoricalActor, self).__init__(name='categorical_actor', **kwargs)
        self.logits_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=dim_obs),
                tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128),
                tf.keras.layers.Dense(num_act)
            ]
        )

    def _distribution(self, obs):
        logits = self.logits_net(obs)

        return tfd.Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def call(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, np.squeeze(act))

        return pi, logp_a

class GaussianActor(tf.keras.Model):

    def __init__(self, dim_obs, dim_act, **kwargs):
        super(GaussianActor, self).__init__(name='gaussian_actor', **kwargs)
        self.mean_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=dim_obs),
                tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128),
                tf.keras.layers.Dense(dim_act)
            ]
        )
        self.log_std = tf.Variable(initial_value=-0.5*np.ones(dim_act, dtype=np.float32))

    def _distribution(self, obs):
        mean = tf.squeeze(self.mu_net(obs))
        std = tf.math.exp(self.log_std)

        return tfd.Normal(loc=mean, scale=std)

    def _log_prob_from_distribution(self, pi, act):
        return tf.math.reduce_sum(pi.log_prob(act), axis=-1)

    def call(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)

        return pi, logp_a

class Critic(tf.keras.Model):

    def __init__(self, dim_obs, **kwargs):
        super(Critic, self).__init__(name='critic', **kwargs)
        self.val_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=dim_obs),
                tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128),
                tf.keras.layers.Dense(1)
            ]
        )

    @tf.function
    def call(self, obs):
        return tf.squeeze(self.val_net(obs), axis=-1)

class Encoder(tf.keras.Model):
    """
    Encode image into Gaussian distributions
    """

    def __init__(self, dim_latent, dim_origin, **kwargs):
        super(Encoder, self).__init__(name='encoder', **kwargs)
        self.dim_latent = dim_latent # scalar
        self.dim_origin = dim_origin # (x,y,z)
        # construct encoder
        inputs_img = tf.keras.Input(shape=dim_origin)
        features_conv = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(inputs_img)
        features_conv = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(features_conv)
        features_conv = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(features_conv)
        features_dense = tf.keras.layers.Flatten()(features_conv)
        outputs_mean = tf.keras.layers.Dense(dim_latent)(features_dense)
        outputs_logstd = tf.keras.layers.Dense(dim_latent)(features_dense)
        self.encoder = tf.keras.Model(inputs=inputs_img, outputs = [outputs_mean, outputs_logstd])
        
    @tf.function
    def call(self, x):
        mean, logstd = self.encoder(x)
        return mean, logstd

class Decoder(tf.keras.Model):
    """
    Decode Gaussian distributions to image
    """

    def __init__(self, dim_latent, **kwargs):
        super(Decoder, self).__init__(name='decoder', **kwargs)
        self.dim_latent = dim_latent # scalar
        # construct decoder
        inputs_latent = tf.keras.Input(shape=(dim_latent,))
        features_dense = tf.keras.layers.Dense(units=16*16*32, activation='relu')(inputs_latent)
        features_conv = tf.keras.layers.Reshape(target_shape=(16, 16, 32))(features_dense)
        features_conv = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(features_conv)
        features_conv = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(features_conv)
        features_conv = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(features_conv)
        outputs_img = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')(features_conv)
        self.decoder = tf.keras.Model(inputs=inputs_latent, outputs=outputs_img)
        
    @tf.function
    def call(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.math.sigmoid(logits)
            return probs
        return logits


class IntrinsicMotivationAgent(tf.keras.Model):

    def __init__(self, dim_latent, dim_origin, act_type, dim_obs, dim_act, num_act=None, clip_ratio=0.2, beta=0., target_kl=0.01, **kwargs):
        super(IntrinsicMotivationAgent, self).__init__(name='ppo', **kwargs)
        self.dim_latent = dim_latent
        self.dim_origin = dim_origin
        self.act_type = act_type
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.clip_ratio = clip_ratio
        self.beta = beta
        self.target_kl = target_kl
        if act_type == 'discrete':
            self.actor = CategoricalActor(dim_obs, num_act)
        elif act_type == 'continuous':
            self.actor = GaussianActor(dim_obs, dim_act)
        self.critic = Critic(dim_obs)
        # self.autoencoder = CVAE(dim_latent=dim_latent, dim_origin=dim_origin)
        self.encoder = Encoder(dim_latent=dim_latent, dim_origin=dim_origin)
        self.decoder = Decoder(dim_latent=dim_latent)
        self.imaginator = Encoder(dim_latent=dim_latent, dim_origin=dim_origin)
        weights_share = self.imaginator.get_weights()
        weights_share[:-4] = self.encoder.get_weights()[:-4] # last 4 layers are dense connections from conv features to mean and logstd plus biases.
        self.imaginator.set_weights(weights_share)
        # self.imagination = tfd.Normal(loc=tf.zeros(dim_latent), scale=tf.zeros(dim_latent))
        # self.prev_kld = tf.Variable(0.)
        self.optimizer_vae = tf.keras.optimizers.Adam(3e-4)
        self.optimizer_actor = tf.keras.optimizers.Adam(1e-4)
        self.optimizer_critic = tf.keras.optimizers.Adam(3e-4)
        self.optimizer_imaginator = tf.keras.optimizers.Adam(3e-4)

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decoder(eps, apply_sigmoid=True)

    @tf.function
    def reparameterize(self, mean, logstd):
        eps = tf.random.normal(shape=mean.shape)
        return eps*tf.math.exp(logstd) + mean

    def pi_of_a_given_s(self, obs):
        with tf.GradientTape() as t:
            with t.stop_recording():
                pi = self.actor._distribution(obs) # policy distribution (Gaussian)
                act = tf.squeeze(pi.sample())
                logp_a = self.actor._log_prob_from_distribution(pi, act)
                val = tf.squeeze(self.critic(obs), axis=-1)

        return act.numpy(), val.numpy(), logp_a.numpy()

    def imagine(self, image):
        """
        Set a goal and compute KL-Divergence between the imagination and the current state
        """
        imagined_mean, imagined_logstd = self.imaginator(image)
        self.imagination = tfd.Normal(imagined_mean, tf.math.exp(imagined_logstd))
        # sample and decode imagination
        self.imagination_sample = self.reparameterize(imagined_mean, imagined_logstd)
        self.decoded_imagination = self.decoder(self.imagination_sample, apply_sigmoid=True) # just decode 1 sample
        # compute kl-divergence between imagined and encoded state
        encoded_mean, encoded_logstd = self.encoder(image)
        self.encoded_image = tfd.Normal(encoded_mean, tf.math.exp(encoded_logstd))
        self.prev_kld = tf.math.reduce_sum(tfd.kl_divergence(self.imagination, self.encoded_image), axis=-1)

    def compute_intrinsic_reward(self, next_image):
        """
        kld_t - kld_{t+1}
        """
        encoded_mean, encoded_logstd = self.encoder(next_image)
        self.encoded_image = tfd.Normal(encoded_mean, tf.math.exp(encoded_logstd))
        self.kld = tf.math.reduce_sum(tfd.kl_divergence(self.imagination, self.encoded_image), axis=-1)
        reward = self.prev_kld - self.kld
        # self.prev_kld = kld

        return np.squeeze(reward)
        
    def train_autoencoder(self, dataset, num_epochs):

        def log_normal_pdf(sample, mean, logstd, raxis=1):
            return tf.math.reduce_sum(tfd.Normal(loc=mean, scale=tf.math.exp(logstd)).log_prob(sample), axis=-1)
        elbo_per_epoch = []
        for epc in range(num_epochs):
            step_cntr = 0
            metrics_mean = tf.keras.metrics.Mean()
            for batch in dataset:
                with tf.GradientTape() as tape:
                    tape.watch(self.encoder.trainable_weights + self.decoder.trainable_weights)
                    mean, logstd = self.encoder(batch)
                    z = self.reparameterize(mean, logstd)
                    reconstructed = self.decoder(z)
                    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=reconstructed, labels=batch)
                    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
                    logpz = log_normal_pdf(z, 0., 0.)
                    logqz_x = log_normal_pdf(z, mean, logstd)
                    loss_vae = -tf.math.reduce_mean(logpx_z + logpz - logqz_x)
                    loss_vae_mean = metrics_mean(loss_vae)
                gradients_vae = tape.gradient(loss_vae, self.encoder.trainable_weights+self.decoder.trainable_weights)
                self.optimizer_vae.apply_gradients(zip(gradients_vae, self.encoder.trainable_weights+self.decoder.trainable_weights))
                step_cntr += 1
                logging.debug("Epoch: {}, Step: {} \nVAE Loss: {}".format(epc+1, step_cntr, loss_vae))
            elbo = -loss_vae_mean.numpy()
            logging.info("Epoch {} ELBO: {}".format(epc+1,elbo))
            elbo_per_epoch.append(elbo)
        return elbo_per_epoch
                
    def train_policy(self, data, num_iters):

        def normal_entropy(log_std):
            return .5*tf.math.log(2.*np.pi*np.e*tf.math.exp(log_std)**2)

        init_images = data['obs'][0:-1:10, :, :, 0]
        # update actor
        for i in range(num_iters):
            logging.debug("Staring actor epoch: {}".format(i+1))
            ep_kl = tf.convert_to_tensor([]) 
            ep_ent = tf.convert_to_tensor([]) 
            with tf.GradientTape() as tape:
                tape.watch(self.actor.trainable_variables)
                mean, logstd = self.imaginator(np.expand_dims(init_images, -1))
                z = self.reparameterize(mean, logstd)
                imgntn_dec = self.decoder(z)
                obs_rec = np.zeros((data['obs'].shape[0], data['obs'].shape[1], data['obs'].shape[2], 1))
                for j in range(30):
                    for k in range(10):
                        obs_rec[j*10+k,:,:,:] = imgntn_dec[j,:,:,:]
                obs = np.concatenate((np.expand_dims(data['obs'][:,:,:,0], -1), obs_rec), -1)
                # logp = self.actor(data['obs'], data['act']) 
                pi, logp = self.actor(obs, data['act']) 
                ratio = tf.math.exp(logp - data['logp']) # pi/old_pi
                clip_adv = tf.math.multiply(tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio), data['adv'])
                approx_kl = tf.reshape(data['logp'] - logp, shape=[-1])
                ent = tf.reshape(tf.math.reduce_sum(pi.entropy(), axis=-1), shape=[-1])
                obj = tf.math.minimum(tf.math.multiply(ratio, data['adv']), clip_adv) + self.beta*ent
                loss_pi = -tf.math.reduce_mean(obj)
            # gradient descent actor weights
            grads_actor = tape.gradient(loss_pi, self.actor.trainable_variables)
            self.optimizer_actor.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
            # record kl-divergence and entropy
            ep_kl = tf.concat([ep_kl, approx_kl], axis=0)
            ep_ent = tf.concat([ep_ent, ent], axis=0)
            # log epoch
            kl = tf.math.reduce_mean(ep_kl)
            entropy = tf.math.reduce_mean(ep_ent)
            logging.info("Epoch :{} \nLoss: {} \nEntropy: {} \nKLDivergence: {}".format(
                i+1,
                loss_pi,
                entropy,
                kl
            ))
            # early cutoff due to large kl-divergence
            # if kl > 1.5*self.target_kl:
            #     logging.warning("Early stopping at epoch {} due to reaching max kl-divergence.".format(epch+1))
            #     break
        # update critic
        for i in range(num_iters):
            logging.debug("Starting critic epoch: {}".format(i))
            with tf.GradientTape() as tape:
                tape.watch(self.critic.trainable_variables)
                loss_v = tf.keras.losses.MSE(data['ret'], self.critic(data['obs']))
            # gradient descent critic weights
            grads_critic = tape.gradient(loss_v, self.critic.trainable_variables)
            self.optimizer_critic.apply_gradients(zip(grads_critic, self.critic.trainable_variables))
            # log epoch
            logging.info("Epoch :{} \nLoss: {}".format(
                i+1,
                loss_v
            ))

        return loss_pi, loss_v, dict(kl=kl, ent=entropy) 
