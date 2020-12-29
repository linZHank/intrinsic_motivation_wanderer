import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from agents.intrinsic_motivation_agent import IntrinsicMotivationAgent

dim_latent = 8
dim_view = (128,128,1)
dim_act = 1
num_act = 10

agent = IntrinsicMotivationAgent(dim_latent,dim_view,dim_act,num_act)
