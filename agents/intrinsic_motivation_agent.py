"""
Intrinsic reward driven agent class using image as input
"""
import tensorflow as tf
import numpy as np
import scipy.signal
import tensorflow_probability as tfp
tfd = tfp.distributions


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

    def __init__(self, dim_act, size, gamma=0.99, lam=0.95):
        # self.obs_buf = np.zeros([size]+list(dim_obs), dtype=np.float32)
        # self.obs_buf = np.zeros((size, dim_obs), dtype=np.float32)
        self.act_buf = np.zeros((size, dim_act), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, act, rew, val, logp):
        assert self.ptr <= self.max_size     # buffer has to have room so you can store
        # self.obs_buf[self.ptr] = obs
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
        data = dict(act=self.act_buf, ret=self.ret_buf,
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
                tf.keras.layers.Dense(1)
            ]
        )

    @tf.function
    def call(self, obs):
        return tf.squeeze(self.val_net(obs), axis=-1)

class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, dim_latent, dim_obs):
        super(CVAE, self).__init__()
        self.dim_latent = dim_latent
        self.dim_obs = dim_obs
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=dim_obs),
                tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(dim_latent + dim_latent),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(dim_latent,)),
                tf.keras.layers.Dense(units=16*16*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(16, 16, 32)),
                tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.dim_latent))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


class IntrinsicMotivationAgent(tf.keras.Model):

    def __init__(self, act_type, dim_latent, dim_obs, dim_act, num_act=None, **kwargs):
        super(IntrinsicMotivationAgent, self).__init__(name='ppo', **kwargs)
        self.act_type = act_type
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        if act_type == 'discrete':
            self.actor = CategoricalActor(dim_obs, num_act)
        elif act_type == 'continuous':
            self.actor = GaussianActor(dim_obs, dim_act)
        self.critic = Critic(dim_obs)
        self.autoencoder = CVAE(dim_latent=dim_latent, dim_obs = dim_obs)
        self.imaginator = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=dim_obs),
                tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(dim_latent + dim_latent),
            ]
        )

    def pi_of_a_given_s(self, obs):
        with tf.GradientTape() as t:
            with t.stop_recording():
                pi = self.actor._distribution(obs) # policy distribution (Gaussian)
                act = tf.squeeze(pi.sample())
                logp_a = self.actor._log_prob_from_distribution(pi, act)
                val = tf.squeeze(self.critic(obs), axis=-1)

        return act.numpy(), val.numpy(), logp_a.numpy()

    def imagine(self, obs):
        mean, logvar = tf.split(self.imaginator(obs), num_or_size_splits=2, axis=1)
        return mean, logvar
