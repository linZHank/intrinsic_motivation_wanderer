"""
Intrinsic reward driven agent class using image as input
"""
import tensorflow as tf
import numpy as np
import scipy.signal
import tensorflow_probability as tfp
tfd = tfp.distributions
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.DEBUG)

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

class OnPolicyBuffer: # To save memory, no image will be saved. 

    def __init__(self, dim_obs=8, dim_act=1, max_size=1000, gamma=0.99, lam=0.97):
        # params
        self.dim_state = dim_obs
        self.dim_act = dim_act
        self.max_size = max_size
        self.gamma = gamma
        self.lam = lam
        # init buffers
        self.obs_buf = np.zeros(shape=(max_size, dim_obs), dtype=np.float32) # state, default dtype=tf.float32
        self.act_buf = np.zeros(shape=(max_size, dim_act), dtype=np.float32) # action
        self.rew_buf = np.zeros(shape=(max_size,), dtype=np.float32) # reward
        self.val_buf = np.zeros(shape=(max_size,), dtype=np.float32) # value
        self.ret_buf = np.zeros(shape=(max_size,), dtype=np.float32) # rewards-to-go return
        self.adv_buf = np.zeros(shape=(max_size,), dtype=np.float32) # advantage
        self.lpa_buf = np.zeros(shape=(max_size,), dtype=np.float32) # logprob(action)
        # variables
        self.ptr, self.path_start_idx = 0, 0

    def store(self, obs, act, rew, val, lpa):
        assert self.ptr <= self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
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
        self.obs_buf = self.obs_buf[:self.ptr] 
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
            obs=self.obs_buf, 
            act=self.act_buf, 
            ret=self.ret_buf, 
            adv=self.adv_buf, 
            lpa=self.lpa_buf
        )
        return {k: tf.convert_to_tensor(v, dtype=tf.float32) for k,v in data.items()}
################################################################

################################################################
class Encoder(tf.keras.Model):
    def __init__(self, dim_origin, dim_latent, **kwargs):
        super(Encoder, self).__init__(name='encoder', **kwargs)
        # params
        self.dim_origin = dim_origin
        self.dim_latent = dim_latent
        # encoder net
        inputs = tf.keras.Input(shape=dim_origin, name='encoder_inputs')
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation='relu')(inputs)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        mu = tf.keras.layers.Dense(dim_latent, name='encoder_outputs_mean')(x)
        logsigma = tf.keras.layers.Dense(dim_latent, name='encoder_outputs_logstddev')(x)
        self.encoder_net = tf.keras.Model(inputs=inputs, outputs = [mu, logsigma])

    @tf.function
    def call(self, origin):
        mean, logstddev = self.encoder_net(origin)
        return tf.squeeze(mean), tf.squeeze(logstddev)

class Decoder(tf.keras.Model):
    def __init__(self, dim_latent, **kwargs):
        super(Decoder, self).__init__(name='decoder', **kwargs)
        # params
        self.dim_latent = dim_latent
        # encoder net
        inputs = tf.keras.Input(shape=dim_latent, name='encoder_inputs')
        x = tf.keras.layers.Dense(units=16*16*64, activation='relu')(inputs) 
        x = tf.keras.layers.Reshape(target_shape=(16, 16, 64))(x)
        x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', activation='relu')(x)
        outputs = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=4, strides=1, padding='same', name='decoder_outputs')(x)
        self.decoder_net = tf.keras.Model(inputs=inputs, outputs = outputs)

    @tf.function
    def call(self, latent, apply_sigmoid=False):
        reconstruction = self.decoder_net(latent)
        if apply_sigmoid:
            probs = tf.math.sigmoid(reconstruction)
            return probs
        return reconstruction

class VariationalAutoencoder(tf.keras.Model):
    """
    Variational autoencoder
    """
    def __init__(self, dim_origin, dim_latent, **kwargs):
        super(VariationalAutoencoder, self).__init__(name='vae', **kwargs)
        # params
        self.dim_origin = dim_origin
        self.dim_latent = dim_latent
        # modules
        self.encoder = Encoder(dim_origin, dim_latent)
        self.decoder = Decoder(dim_latent)
        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

    @tf.function
    def reparameterize(self, mean, logstddev):
        eps = tf.random.normal(shape=mean.shape)
        return mean + eps*tf.math.exp(logstddev) 

    def train(self, dataset, num_epochs):

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
                    mean, logstd = self.encoder(batch)
                    z = self.reparameterize(mean, logstd)
                    x_rec = self.decoder(z)
                    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_rec, labels=batch)
                    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
                    logpz = log_normal_pdf(z, 0., 0.)
                    logqz_x = log_normal_pdf(z, mean, logstd)
                    loss_vae = -tf.math.reduce_mean(logpx_z + logpz - logqz_x)
                    loss_vae_mean = metrics_mean(loss_vae)
                gradients_vae = tape.gradient(loss_vae, self.encoder.trainable_weights+self.decoder.trainable_weights)
                self.optimizer.apply_gradients(zip(gradients_vae, self.encoder.trainable_weights+self.decoder.trainable_weights))
                step_cntr += 1
                logging.debug("Epoch: {}, Step: {} \nVAE Loss: {}".format(e+1, step_cntr, loss_vae))
            elbo = -loss_vae_mean
            logging.info("Epoch {} ELBO: {}".format(e+1,elbo))
            elbo_per_epoch.append(elbo)
        return elbo_per_epoch
################################################################

################################################################
class CategoricalActor(tf.keras.Model):

    def __init__(self, dim_obs, num_act, **kwargs):
        super(CategoricalActor, self).__init__(name='categorical_actor', **kwargs)
        self.dim_obs=dim_obs
        self.num_act=num_act
        self.policy_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(dim_obs,), name='actor_inputs'),
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
                tf.keras.layers.InputLayer(input_shape=(dim_obs,), name='critic_inputs'),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(1, activation=None, name='critic_outputs')
            ]
        )

    @tf.function
    def call(self, obs):
        return tf.squeeze(self.value_net(obs))

class PPOActorCritic(tf.keras.Model):
    def __init__(self, dim_obs=8, num_act=10, dim_act=1, clip_ratio=0.2, beta=0., target_kld=0.1, **kwargs):
        super(PPOActorCritic, self).__init__(name='ppo_ac', **kwargs)
        # parameters
        self.dim_obs = dim_obs
        self.num_act = num_act
        self.dim_act = dim_act
        self.clip_ratio = clip_ratio
        self.beta = beta
        self.target_kld = target_kld
        # modules
        self.actor = CategoricalActor(dim_obs, num_act)
        self.critic = Critic(dim_obs)
        # optimizers
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=3e-4)

    @tf.function
    def make_decision(self, obs):
        pi = self.actor._distribution(obs)
        act = pi.sample()
        logp_a = self.actor._logprob(pi, act)
        val = self.critic(obs)

        return act, val, logp_a

    def train(self, data, num_epochs):
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
################################################################

################################################################
class DynamicsModel(tf.keras.Model):
    def __init__(self, dim_obs, dim_act, **kwargs):
        super(DynamicsModel, self).__init__(name='dynamics_model', **kwargs)
        # params
        self.dim_obs=dim_obs
        self.dim_act=dim_act
        # model
        inputs_obs = tf.keras.Input(shape=(dim_obs,), name='dynamics_inputs_obs')
        inputs_act = tf.keras.Input(shape=(dim_act,), name='dynamics_inputs_act')
        x = tf.keras.layers.concatenate([inputs_obs, inputs_act])
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        outputs_mean = tf.keras.layers.Dense(dim_obs, name='dynamics_ouputs_mean')(x)
        outputs_logstddev = tf.keras.layers.Dense(dim_obs, name='dynamics_ouputs_logstddev')(x)
        self.dynamics_net = tf.keras.Model(inputs=[inputs_obs, inputs_act], outputs=[outputs_mean, outputs_logstddev])
        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

    @tf.function
    def call(self, obs, act):
        mean, logstddev = self.dynamics_net([obs, act])
        return tf.squeeze(mean), tf.squeeze(logstddev)

    @tf.function
    def reparameterize(self, mean, logstddev):
        eps = tf.random.normal(shape=mean.shape)
        return mean + eps*tf.math.exp(logstddev) 

    def train(self, data, num_epoch):
        ep_loss_dyna = []
        for e in range(num_epochs):
            logging.debug("Starting dynamics model training epoch: {}".format(e+1))
            with tf.GradientTape() as tape:
                tape.watch(self.dynamics_net.trainable_variables)
                z_true = self.reparameterize(data['encmu'], data['encls'])
                mu_pred, logsigma_pred = self.call(data['obs'], data['act'])
                z_pred = self.reparameterize(mu_pred, tf.math.exp(logsigma_pred))
                loss_dyna = tf.keras.losses.KLD(z_true, z_pred)
            grads_dyna = tape.gradient(loss_dyna, self.dynamics_net.trainable_variables)
            self.optimizer.apply_gradients(zip(grads_dyna, self.dynamics_net.trainable_variables))
            ep_loss_dyna.append(loss_dyna)
            logging.debug("\n----Dynamics Model Training----\nEpoch :{} \nLoss: {}".format(
                e+1,
                loss_dyna
            ))
        mean_loss_dyna = tf.math.reduce_mean(ep_loss_dyna)
        
        return mean_loss_dyna

################################################################

class IntrinsicMotivationAgent(tf.keras.Model):
    def __init__(self, dim_view=(128,128,1), dim_latent=8, num_act=10, dim_act=1, clip_ratio=0.2, beta=0., target_kld=0.1, **kwargs):
        super(IntrinsicMotivationAgent, self).__init__(name='ima', **kwargs)
        # parameters
        self.dim_view = dim_view
        self.dim_latent = dim_latent
        self.num_act = num_act
        self.dim_act = dim_act
        self.clip_ratio = clip_ratio
        self.beta = beta
        self.target_kld = target_kld
        # modules
        self.vae = VariationalAutoencoder(dim_view, dim_latent)
        self.ac = PPOActorCritic(dim_obs=dim_latent, num_act=num_act, dim_act=dim_act, clip_ratio=clip_ratio, beta=beta, target_kld=target_kld)
        self.imaginator = DynamicsModel(dim_latent, dim_act)

    @ tf.function
    def compute_intrinsic_reward(self, imagine_mean, imagine_logstddev, latent_feature):
        """
        Less likely state results in larger reward, which simulates curiosity
        """
        imagine_distribution = tfd.Normal(loc=imagine_mean, scale=tf.math.exp(imagine_logstddev))
        reward = -tf.math.reduce_mean(imagine_distribution.log_prob(latent_feature), axis=-1)

        return tf.squeeze(tf.clip_by_value(reward, 0, 10))

# class IntrinsicMotivationAgent(tf.keras.Model):
#     def __init__(self, dim_view=(128,128,1), dim_latent=8, num_act=10, dim_act=1, clip_ratio=0.2, beta=0., target_kld=0.1, **kwargs):
#         super(IntrinsicMotivationAgent, self).__init__(name='ima', **kwargs)
#         # parameters
#         self.dim_view = dim_view
#         self.dim_latent = dim_latent
#         self.dim_act = dim_act
#         self.clip_ratio = clip_ratio
#         self.beta = beta
#         self.target_kld = target_kld
#         # modules
#         self.vae = VAE(dim_view, dim_latent)
#         self.actor = CategoricalActor(dim_latent, num_act)
#         self.critic = Critic(dim_latent)
#         self.imaginator = DynamicsModel(dim_latent, dim_act)
#         # optimizers
#         self.optimizer_vae = tf.keras.optimizers.Adam(learning_rate=3e-4)
#         self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=3e-4)
#         self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=3e-4)
#         self.optimizer_imaginator = tf.keras.optimizers.Adam(learning_rate=3e-4)
# 
#     @tf.function
#     def make_decision(self, obs):
#         pi = self.actor._distribution(obs)
#         act = pi.sample()
#         logp_a = self.actor._logprob(pi, act)
#         val = self.critic(obs)
# 
#         return act, val, logp_a
# 
#     @ tf.function
#     def compute_intrinsic_reward(self, imagine_mean, imagine_logstddev, latent_feature):
#         """
#         Less likely state results in larger reward, which simulates curiosity
#         """
#         imagine_distribution = tfd.Normal(loc=imagine_mean, scale=tf.math.exp(imagine_logstddev))
#         reward = -tf.math.reduce_mean(imagine_distribution.log_prob(latent_feature), axis=-1)
# 
#         return tf.squeeze(tf.clip_by_value(reward, 0, 10))
# 
#     def train_vae(self, dataset, num_epochs):
# 
#         def log_normal_pdf(sample, mean, logstd, raxis=1):
#             log_pi = tf.math.log(2.*np.pi)
#             return tf.math.reduce_sum(-.5*(((sample-mean)*tf.math.exp(-logstd))**2 + 2*logstd + log_pi), axis=-1)
#         elbo_per_epoch = []
#         for e in range(num_epochs):
#             step_cntr = 0
#             metrics_mean = tf.keras.metrics.Mean()
#             for batch in dataset:
#                 with tf.GradientTape() as tape:
#                     tape.watch(self.encoder.trainable_weights + self.decoder.trainable_weights)
#                     mean, logstd = self.encode(batch)
#                     z = self.reparameterize(mean, logstd)
#                     reconstructed = self.decode(z)
#                     cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=reconstructed, labels=batch)
#                     logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
#                     logpz = log_normal_pdf(z, 0., 0.)
#                     logqz_x = log_normal_pdf(z, mean, logstd)
#                     loss_vae = -tf.math.reduce_mean(logpx_z + logpz - logqz_x)
#                     loss_vae_mean = metrics_mean(loss_vae)
#                 gradients_vae = tape.gradient(loss_vae, self.encoder.trainable_weights+self.decoder.trainable_weights)
#                 self.optimizer_vae.apply_gradients(zip(gradients_vae, self.encoder.trainable_weights+self.decoder.trainable_weights))
#                 step_cntr += 1
#                 logging.debug("Epoch: {}, Step: {} \nVAE Loss: {}".format(e+1, step_cntr, loss_vae))
#             elbo = -loss_vae_mean
#             logging.info("Epoch {} ELBO: {}".format(e+1,elbo))
#             elbo_per_epoch.append(elbo)
#         return elbo_per_epoch
# 
#     def train_ac(self, data, num_epochs):
#         # Update actor
#         ep_loss_pi = []
#         ep_kld = []
#         ep_ent = []
#         for e in range(num_epochs):
#             logging.debug("Staring actor training epoch: {}".format(e+1))
#             with tf.GradientTape() as tape:
#                 tape.watch(self.actor.trainable_variables)
#                 pi, lpa = self.actor(data['obs'], data['act'])
#                 ratio = tf.math.exp(lpa - data['lpa']) # pi/old_pi
#                 clip_adv = tf.math.multiply(tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio), data['adv'])
#                 approx_kld = data['lpa'] - lpa
#                 ent = tf.math.reduce_sum(pi.entropy(), axis=-1)
#                 obj = tf.math.minimum(ratio*data['adv'], clip_adv) + self.beta*ent
#                 loss_pi = -tf.math.reduce_mean(obj)
#             # gradient descent actor weights
#             grads_actor = tape.gradient(loss_pi, self.actor.trainable_variables)
#             self.optimizer_actor.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
#             ep_loss_pi.append(loss_pi)
#             ep_kld.append(tf.math.reduce_mean(approx_kld))
#             ep_ent.append(tf.math.reduce_mean(ent))
#             # log epoch
#             logging.info("\n----Actor Training----\nEpoch :{} \nLoss: {} \nKLDivergence: {} \nEntropy: {}".format(
#                 e+1,
#                 loss_pi,
#                 ep_kld[-1],
#                 ep_ent[-1],
#             ))
#             # early cutoff due to large kl-divergence
#             if ep_kld[-1] > self.target_kld:
#                 logging.warning("\nEarly stopping at epoch {} due to reaching max kl-divergence.\n".format(e+1))
#                 break
#         mean_loss_pi = tf.math.reduce_mean(ep_loss_pi)
#         mean_ent = tf.math.reduce_mean(ep_ent)
#         mean_kld = tf.math.reduce_mean(ep_kld)
#         # Update critic
#         ep_loss_val = []
#         for e in range(num_epochs):
#             logging.debug("Starting critic training epoch: {}".format(e+1))
#             with tf.GradientTape() as tape:
#                 tape.watch(self.critic.trainable_variables)
#                 # loss_val = self.mse(data['ret'], self.critic(data['obs']))
#                 loss_val = tf.keras.losses.MSE(data['ret'], self.critic(data['obs']))
#             # gradient descent critic weights
#             grads_critic = tape.gradient(loss_val, self.critic.trainable_variables)
#             self.optimizer_critic.apply_gradients(zip(grads_critic, self.critic.trainable_variables))
#             ep_loss_val.append(loss_val)
#             # log loss_v
#             logging.info("\n----Critic Training----\nEpoch :{} \nLoss: {}".format(
#                 e+1,
#                 loss_val
#             ))
#         mean_loss_val = tf.math.reduce_mean(ep_loss_val)
# 
#         return mean_loss_pi, mean_loss_val, dict(kld=mean_kld, entropy=mean_ent)

    


# data_dir = '/media/palebluedotian0/Micron1100_2T/playground/intrinsic_motivation_wanderer/experience/2021-01-20-17-07/'
# dataset = tf.keras.preprocessing.image_dataset_from_directory(
#     data_dir,
#     color_mode='grayscale',
#     image_size=(128, 128),
#     batch_size=64
# )
# dataset = dataset.map(lambda x, y: x/254.)
# vae = VAE()
# loss_elbo = vae.train(dataset, num_epochs=20)
# import os
# import matplotlib.pyplot as plt
# import cv2
# sf = np.load(os.path.join(data_dir, 'stepwise_frames.npy'))
# frames = sf[-11:]-1
# fig, ax = plt.subplots(nrows=10, ncols=2)
# for i in range(len(frames)-1):
#     # original 
#     ori = cv2.imread(os.path.join(data_dir, 'views', str(frames[i])+'.jpg'), 0)/254.
#     ax[i,0].imshow(ori, cmap='gray')
#     ax[i,0].axis('off')
#     # reconstruction
#     mu, logsigma = vae.encode(np.expand_dims(ori,0))
#     z = vae.reparameterize(mu, tf.math.exp(logsigma))
#     rec = vae.decode(z)
#     ax[i,1].imshow(rec[0,:,:,0], cmap='gray')
#     ax[i,1].axis('off')
# 
# plt.subplots_adjust(wspace=0, hspace=.1)
# plt.tight_layout()
# plt.show()

