#!/usr/bin/python3
"""
Initialize Intrinsic Motivation Agent (IMA)
"""
import sys
import os
from datetime import datetime

from agents.intrinsic_motivation_agent import IntrinsicMotivationAgent


# Parameters
dim_latent = 16
dim_view = (128,128,1)
dim_act = 1
num_act = 10
brain = IntrinsicMotivationAgent(dim_latent=dim_latent, dim_view=dim_view, dim_act=dim_act, num_act=num_act)
seeds = ['belauensis', 'macromphalus', 'pompilius', 'scrobiculatus', 'stenomphalus'] # For memorization of the vanished pattern in front of the Rhodes Hall. Every new born agent will be given a name after a valid species in the family of Nautilidae listed in https://en.wikipedia.org/wiki/Nautilus_(genus)

# Set model paths
index = 0 
model_dir = os.path.join(sys.path[0], 'model_dir', seeds[index], datetime.now().strftime("%Y-%m-%d-%H-%M"))
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
encoder_path = os.path.join(model_dir, 'encoder')
decoder_path = os.path.join(model_dir, 'decoder')
imaginator_path = os.path.join(model_dir, 'imaginator')
actor_path = os.path.join(model_dir, 'actor')
critic_path = os.path.join(model_dir, 'critic')
# Save models
brain.encoder.save(encoder_path)
brain.decoder.save(decoder_path)
brain.imaginator.save(imaginator_path)
brain.actor.save(actor_path)
brain.critic.save(critic_path)
