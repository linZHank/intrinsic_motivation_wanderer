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

    def __init__(self, dim_latent, dim_act, size, gamma=0.99, lam=0.95):
        self.latent_mean_buf = np.zeros((size, dim_latent), dtype=np.float32)
        self.latent_stddev_buf = np.zeros((size, dim_latent), dtype=np.float32)
        self.imagination_mean_buf = np.zeros((size, dim_latent), dtype=np.float32)
        self.imagination_stddev_buf = np.zeros((size, dim_latent), dtype=np.float32)
        self.act_buf = np.zeros((size, dim_act), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, latent_mean, latent_stddev, imagination_mean, imagination_stddev, act, rew, val, logp):
        assert self.ptr <= self.max_size     # buffer has to have room so you can store
        self.latent_mean_buf[self.ptr] = latent_mean
        self.latent_stddev_buf[self.ptr] = latent_stddev
        self.imagination_mean_buf[self.ptr] = imagination_mean
        self.imagination_stddev_buf[self.ptr] = imagination_stddev
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
        data = dict(
            latent_mean=self.latent_mean_buf, 
            latent_stddev=self.latent_stddev_buf, 
            imagination_mean=self.imagination_mean_buf, 
            imagination_stddev=self.imagination_stddev_buf, 
            act=self.act_buf, 
            ret=self.ret_buf, 
            adv=self.adv_buf, 
            logp=self.logp_buf
        )
        return {k: tf.convert_to_tensor(v, dtype=tf.float32) for k,v in data.items()}
################################################################

class IntrinsicMotivationAgent(tf.keras.Model):

    def __init__(self, dim_latent, dim_view, dim_act, num_act=None, clip_ratio=0.2, beta=0., target_kl=0.01, **kwargs):
        super(IntrinsicMotivationAgent, self).__init__(name='ppo_ima', **kwargs)
        # parameters
        self.dim_latent = dim_latent
        self.dim_view = dim_view
        self.dim_act = dim_act
        self.clip_ratio = clip_ratio
        self.beta = beta
        self.target_kl = target_kl

        # construct encoder
        inputs = tf.keras.Input(shape=dim_view, name='image_input')
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(inputs)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        outputs_mean = tf.keras.layers.Dense(dim_latent, name='encoded_mean')(x)
        outputs_logstd = tf.keras.layers.Dense(dim_latent, name='encoded_logstd')(x)
        self.encoder = tf.keras.Model(inputs=inputs, outputs = [outputs_mean, outputs_logstd])

        # construct decoder
        inputs_latent = tf.keras.Input(shape=(dim_latent,), name='latent_feature')
        x = tf.keras.layers.Dense(units=16*16*32, activation='relu')(inputs_latent) # only valid for reconstructing (128,128,1) image; TODO: generalize to any shape
        x = tf.keras.layers.Reshape(target_shape=(16, 16, 32))(x)
        x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        outputs_img = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same', name='decoded_image')(x)
        self.decoder = tf.keras.Model(inputs=inputs_latent, outputs=outputs_img)

        # construct imaginator
        inputs_mean = tf.keras.Input(shape=dim_latent, name='imaginator_input_mean')
        inputs_stddev = tf.keras.Input(shape=dim_latent, name='imaginator_input_stddev')
        inputs_act = tf.keras.Input(shape=dim_act, name='imaginator_input_act')
        x = tf.keras.layers.concatenate([inputs_mean, inputs_stddev, inputs_act])
        x = tf.keras.layers.Dense(dim_latent*2, activation='relu')(x)
        outputs_mean = tf.keras.layers.Dense(dim_latent, name='imagined_mean')(x)
        outputs_logstd = tf.keras.layers.Dense(dim_latent, name='imagined_logstd')(x)
        self.imaginator = tf.keras.Model(inputs=[inputs_mean, inputs_stddev, inputs_act], outputs=[outputs_mean, outputs_logstd])

        # construct actor
        inputs_mean = tf.keras.Input(shape=dim_latent, name='actor_input_mean')
        inputs_stddev = tf.keras.Input(shape=dim_latent, name='actor_input_stddev')
        x = tf.keras.layers.concatenate([inputs_mean, inputs_stddev])
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        logits = tf.keras.layers.Dense(num_act, activation='tanh', name='act_logits')(x)
        self.actor = tf.keras.Model(inputs=[inputs_mean, inputs_stddev], outputs=logits)

        # construct critic
        inputs_mean = tf.keras.Input(shape=dim_latent, name='critic_input_mean')
        inputs_stddev = tf.keras.Input(shape=dim_latent, name='critic_input_stddev')
        x = tf.keras.layers.concatenate([inputs_mean, inputs_stddev])
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        outputs_value = tf.keras.layers.Dense(1, name='value')(x)
        self.critic = tf.keras.Model(inputs=[inputs_mean, inputs_stddev], outputs=outputs_value)

        # set up optimizers
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

    def encode(self, img):
        mean, logstd = self.encoder(img)
        return tfd.Normal(loc=mean, scale=tf.math.exp(logstd))

    @tf.function
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.math.sigmoid(logits)
            return probs
        return logits

    def imagine(self, latent_distribution, act):
        mean = latent_distribution.mean() 
        stddev = latent_distribution.stddev()
        mean_imgn, logstd_imgn = self.imaginator([mean, stddev, act])
        return tfd.Normal(loc=mean_imgn, scale=tf.math.exp(logstd_imgn))

    def compute_value(self, latent_distribution):
        mean = latent_distribution.mean() 
        stddev = latent_distribution.stddev()
        return tf.squeeze(self.critic([mean, stddev]), axis=-1)

    def make_decision(self, latent_distribution):
        mean = latent_distribution.mean() 
        stddev = latent_distribution.stddev()
        pi = tfd.Categorical(logits=self.actor([mean, stddev]))
        act = tf.squeeze(pi.sample())
        logp_a = pi.log_prob(act)
        val = self.compute_value(latent_distribution)

        return act.numpy(), val.numpy(), logp_a.numpy()

    def compute_intrinsic_reward(self, latent_distribution, imagined_distribution, next_latent_distribution):
        """
        kld_t - kld_{t+1}
        """
        kld_curr = tf.math.reduce_sum(tfd.kl_divergence(imagined_distribution, latent_distribution), axis=-1)
        kld_next = tf.math.reduce_sum(tfd.kl_divergence(imagined_distribution, next_latent_distribution), axis=-1)
        reward = kld_curr - kld_next

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
        # update actor
        for i in range(num_iters):
            logging.debug("Staring actor epoch: {}".format(i+1))
            ep_kl = tf.convert_to_tensor([]) 
            ep_ent = tf.convert_to_tensor([]) 
            with tf.GradientTape() as tape:
                tape.watch(self.actor.trainable_weights)
                pi = tfd.Categorical(logits=self.actor([data['latent_mean'], data['latent_stddev']]))
                logp = pi.log_prob(data['act'])
                ratio = tf.math.exp(logp - data['logp']) # pi/old_pi
                clip_adv = tf.math.multiply(tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio), data['adv'])
                approx_kl = tf.reshape(data['logp'] - logp, shape=[-1])
                ent = tf.reshape(tf.math.reduce_sum(pi.entropy(), axis=-1), shape=[-1])
                obj = tf.math.minimum(tf.math.multiply(ratio, data['adv']), clip_adv) + self.beta*ent
                loss_pi = -tf.math.reduce_mean(obj)
            # gradient descent actor weights
            grads_actor = tape.gradient(loss_pi, self.actor.trainable_weights + self.imaginator.trainable_weights) 
            self.optimizer_actor.apply_gradients(zip(grads_actor, self.actor.trainable_weights + self.imaginator.trainable_weights))
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
                loss_v = tf.keras.losses.MSE(data['ret'], self.critic(data['state']))
            # gradient descent critic weights
            grads_critic = tape.gradient(loss_v, self.critic.trainable_variables)
            self.optimizer_critic.apply_gradients(zip(grads_critic, self.critic.trainable_variables))
            # log epoch
            logging.info("Epoch :{} \nLoss: {}".format(
                i+1,
                loss_v
            ))

        return loss_pi, loss_v, dict(kl=kl, ent=entropy) 

    def train_imaginator(self, data, num_iters):
        for i in range(num_iters):
            logging.debug("Starting imaginator iter: {}".format(i))
            ep_kl = tf.convert_to_tensor([])
            with tf.GradientTape() as tape:
                tape.watch(self.imaginator.trainable_variables)
                mean_imgn, logstd_imgn = self.imaginator([data['latent_mean'], data['latent_stddev'], data['act']])
                distr_ltnt = tfd.Normal(loc=data['latent_mean'], scale=data['latent_stddev'])
                distr_imgn = tfd.Normal(loc=mean_imgn, scale=tf.math.exp(logstd_imgn))
                kld = tf.math.reduce_sum(tfd.kl_divergence(distr_imgn, distr_ltnt), axis=-1)
                loss_i = -tf.math.reduce_mean(kld)
            # gradient descent critic weights
            grads_imaginator = tape.gradient(loss_i, self.imaginator.trainable_variables)
            self.optimizer_imaginator.apply_gradients(zip(grads_critic, self.imaginator.trainable_variables))
            ep_kl = tf.concat([ep_kl, kld], axis=0)
            # log epoch
            kl = tf.math.reduce_mean(ep_kl)
            logging.info("Epoch :{} \nLoss: {}, \nKL-Divergence: {}".format(
                i+1,
                loss_v,
                kl
            ))
            # early cutoff due to large kl-divergence
            if kl > 10*self.target_kl:
                logging.warning("Imaginator training early stop at iter {} due to reaching max kl-divergence.".format(i+1))
                break

        return loss_i, kl 

