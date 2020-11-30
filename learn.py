#!/usr/bin/python3
"""
Use this script on the computing machine. The robot learns through training.
"""
import os
import time
import numpy as np
import cv2
from datetime import datetime
import tensorflow as tf

from agents.intrinsic_motivation_agent import OnPolicyBuffer, IntrinsicMotivationAgent

# Parameters
height = 128
width = 128
batch_size = 128
# Load new images
image_dir = '/media/palebluedotian0/Micron1100_2T/playground/intrinsic_motivation_wanderer/experience/2020-11-25-16-19/'
image_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory=image_dir,
    color_mode='grayscale',
    image_size=(height, width),
    batch_size=batch_size,
)
image_dataset = image_dataset.map(lambda x, y: x/255.)

# Load new replay buffer
# Train VAE
# Train policy

