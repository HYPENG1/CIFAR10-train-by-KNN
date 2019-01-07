'''-----------------------------------------------------------
                            \\\||///
                           { -    - }
	             [][][_]  [  @    @  ]  [_][][]
--------------------------------------------------------------
Author		:		Yopen Hu
foldername	:		CIFAR10_train
filename	:   	KNN_train_Encapsulation.py
Description	:		object identification in CIFAR10 using 
					KNN algorithm 
Date  	  By   Version          Change Description
==============================================================
18/11/13  HYP    0.0       	 extract data & read data
18/11/14  HYP    1.0    KNN algorithm writing(Wrong Answer)
18/11/15  HYP    2.0    	   KNN algorithm debug
18/11/21  HYP    3.0        KNN algorithm Encapsulation
--------------------------------------------------------------
                                    0ooo
--------------------------ooo0-----(    )---------------------
                         (    )     )  /
                          \  (     (__/
                           \__)
-----------------------------------------------------------'''
# This assignment can reach the 28% accuracy.
import numpy as np 							# numpy toolset
import matplotlib.pyplot as plt 			# plot 2D data
from PIL import Image						# Python Image Library
from scipy.spatial.distance import cdist 	# Calculation of distance
from collections import Counter 			# statistics
import random								# generate the random number
import time									# to calculate the time
						# GPU accelerat


class CIFAR10(object):
	def __init__(self, pickle_loc):
		# Description: initation of class CIFAR
		# input 	 : pickle_loc : the location of batch file
		import pickle
		# keys: [b'batch_label', b'labels', b'data', b'filenames']
		with open(pickle_loc, 'rb') as fo:
			batch = pickle.load(fo, encoding='bytes')

		# keys: [b'num_cases_per_batch', b'label_names', b'num_vis']
		with open('CIFAR10_batches/batches.meta', 'rb') as fo:
			batch_meta = pickle.load(fo, encoding='bytes')
		# data decode:
		# label_names -- a 10-element list which gives meaningful names to the 
		# 				 numeric labels in the labels array described above.
		self.label2name = [label.decode('ascii') for label in batch_meta[b'label_names']]
		# self.dict = batch
		# labels -- a list of 10000 numbers in the range 0-9. The number at index
		# 			i indicates the label of the ith image in the array data.		
		self.label = np.array(batch[b'labels'])
		# data   --	a 10000x3072 numpy array of uint8s. Each row of the array 
		# 			stores a 32x32 colour image. The first 1024 entries contain 
		#			the red channel values, the next 1024 the green, and the 
		#			final 1024 the blue. The image is stored in row-major order,
		#			so that the first 32 entries of the array are the red channel 
		#			values of the first row of the image.		
		self.data = np.array(batch[b'data'])

	def display(self, index):
		# Description: 	Display a picture from CIFAR
		# Input      :	index:	the number of target picture
		picture_data = self.data[index]
		picture_label = self.label[index]
		picture = picture_data.reshape(3,32,32)
		picture = picture.transpose(1, 2, 0)		# Exchange of dimension 1 & 3
		img = Image.fromarray(picture,'RGB')		# numpy -> image
		plt.imshow(img)
		label = self.label2name[picture_label]		# label translation
		plt.title(label)
		plt.show()

	def data2img(self, index):
		# Description: 	Display a picture from CIFAR
		# Input      :	index:	the number of target picture		
		picture_data = self.data[index]
		picture = picture_data.reshape(3,32,32)
		picture = picture.transpose(1, 2, 0)		# Exchange of dimension 1 & 3
		return Image.fromarray(picture,'RGB')		# numpy -> image


################################### load data #############################
batch_1 = 'CIFAR10_batches/data_batch_1'
batch_2 = 'CIFAR10_batches/data_batch_2'
batch_3 = 'CIFAR10_batches/data_batch_3'
batch_4 = 'CIFAR10_batches/data_batch_4'
batch_5 = 'CIFAR10_batches/data_batch_5'
batch_t = 'CIFAR10_batches/test_batch'

cifar10_1 = CIFAR10(batch_1)
cifar10_2 = CIFAR10(batch_2)
cifar10_3 = CIFAR10(batch_3)
cifar10_4 = CIFAR10(batch_4)
cifar10_5 = CIFAR10(batch_5)
cifar10_t = CIFAR10(batch_t)


############################# train & predict data ########################
# Nearest Neighbor Classifier
class NearestNeighbor(object):
	def __init__(self):
		pass

	def train(self, X, y):
		# X is N x D where each row is an example. Y is 1-dimension of size N
		# the nearest neighbor classifier simply remembers all the training data
		self.Xtr = X
		self.ytr = y

	def predict(self, X, kNN_k = 1, L2 = False):
		# X is N x D where each row is an example we wish to predict label for
		num_test = X.shape[0]
		if num_test <= 2 * kNN_k:
			print('train_quantity is too few!')
			exit(1)

		# lets make sure that the output type matches the input type
		Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

		# loop over all test rows
		for i in range(num_test):
			# find the nearest training image to the i'th test image
			if ~ L2:
				distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)	# L1 Distance
			else:
				distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))	# L2 Distance

			cnt = np.zeros((10,1))		# count the output category
			for ii in np.argsort(distances)[0 : kNN_k]:
				cnt[ self.ytr[ii] ] += 1

			Ypred[i] = np.argmax( cnt )

		return Ypred

# data pre-figuration
log_file = open('log_KNN.txt', 'a+')	# file can be write or add.
test_num = 100
Ytr = np.concatenate([cifar10_1.label, cifar10_2.label, cifar10_3.label, cifar10_4.label, cifar10_5.label])
Xtr_rows = np.concatenate([cifar10_1.data, cifar10_2.data, cifar10_3.data, cifar10_4.data, cifar10_5.data])
Yte = cifar10_t.label[range(test_num)]
Xte_rows = cifar10_t.data[range(test_num)]

nn = NearestNeighbor()					# create a Nearest Neighbor classifier class
nn.train(Xtr_rows, Ytr)					# train the classifier on the training images and labels
for kk in range(5, 40):
	time_start = time.time()
	Yte_predict = nn.predict(Xte_rows, kNN_k = kk, L2 = False) 		# predict labels on the test images
	# and now print the classification accuracy, which is the average number
	# of examples that are correctly predicted (i.e. label matches)
	time_end = time.time()
	print( 'accuracy: %f kk = %d test_num = %d' %( np.mean(Yte_predict == Yte), kk, test_num))
	log_file.write( 'accuracy: %f kk = %d test_num = %d time_sec = %d \n' \
							%( np.mean(Yte_predict == Yte), kk, test_num, time_end - time_start))
log_file.close()