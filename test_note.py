# This assignment can reach the 28% accuracy.
import numpy as np 							# numpy toolset
import matplotlib.pyplot as plt 			# plot 2D data
from PIL import Image						# Python Image Library
from scipy.spatial.distance import cdist 	# Calculation of distance
from collections import Counter 			# statistics
import random								# generate the random number
import time									# to calculate the time
# import mxnet as mx						# GPU accelerat
import visdom 								# python -m visdom.server
import torch


print(torch.cuda.is_available())