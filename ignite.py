#!/usr/bin/python3
"""
Initialize Intrinsic Motivation Agent (IMA)
"""
import sys
import os
from datetime import datetime
import time
import numpy as np
import cv2
import tensorflow as tf
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

from agents.ima_macromphalus import IntrinsicMotivationAgent


# Parameters
versions = ['belauensis', 'macromphalus', 'pompilius', 'scrobiculatus', 'stenomphalus'] # For memorization of the vanished pattern in front of the Rhodes Hall. Every new born agent will be given a name after a valid species in the family of Nautilidae listed in https://en.wikipedia.org/wiki/Nautilus_(genus)
dim_view = (128,128,1)
dim_latent = 8
num_act = 10
dim_act = 1
# Instantiate agent
brain = IntrinsicMotivationAgent(dim_view, dim_latent, num_act, dim_act)
# Set model paths
model_dir = os.path.join(sys.path[0], 'model_dir', versions[1], datetime.now().strftime("%Y-%m-%d-%H-%M"))
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
encoder_path = os.path.join(model_dir, 'encoder')
decoder_path = os.path.join(model_dir, 'decoder')
imaginator_path = os.path.join(model_dir, 'imaginator')
actor_path = os.path.join(model_dir, 'actor')
critic_path = os.path.join(model_dir, 'critic')
# Save models
brain.vae.encoder.encoder_net.save(encoder_path)
brain.vae.decoder.decoder_net.save(decoder_path)
brain.imaginator.dynamics_net.save(imaginator_path)
brain.ac.actor.policy_net.save(actor_path)
brain.ac.critic.value_net.save(critic_path)
