'''-----------------------------------------------------------
                            \\\||///
                           { -    - }
	             [][][_]  [  @    @  ]  [_][][]
--------------------------------------------------------------
Author		:		Yopen Hu
foldername	:		CIFAR10_train
filename	:   	KNN_train_Debug.py
Description	:		object identification in CIFAR10 using 
					KNN algorithm 
Date  	  By   Version          Change Description
==============================================================
18/11/13  HYP    0.0       	 extract data & read data
18/11/14  HYP    1.0    KNN algorithm writing(Wrong Answer)
18/11/15  HYP    2.0    	   KNN algorithm debug
--------------------------------------------------------------
                                    0ooo
--------------------------ooo0-----(    )---------------------
                         (    )     )  /
                          \  (     (__/
                           \__)
-----------------------------------------------------------'''
# This assignment can reach the 28% accuracy.
import numpy as np 
import matplotlib.pyplot as plt 			# plot 2D data
from PIL import Image						# Python Image Library
from scipy.spatial.distance import cdist 	# Calculation of distance
from collections import Counter 			# statistics
import random								# generate the random number

################################### load data #############################
#------------------- funcation: unpickle ----------------
# Description: 	Offical data package decode
# Input      :	file:	the location of package
# Output     :	dict:	the data sheet from file
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# For 'CIFAR10_batches/batches.meta':
# label_names -- a 10-element list which gives meaningful names to the 
# 				 numeric labels in the labels array described above. For 
#				 example, label_names[0] == "airplane", label_names[1] == 
#				 "automobile", etc.
# keys: [b'num_cases_per_batch', b'label_names', b'num_vis']
batches_meta = unpickle('CIFAR10_batches/batches.meta')

# For 'CIFAR10_batches/data_batch_X':
# data   --	a 10000x3072 numpy array of uint8s. Each row of the array 
# 			stores a 32x32 colour image. The first 1024 entries contain 
#			the red channel values, the next 1024 the green, and the 
#			final 1024 the blue. The image is stored in row-major order,
#			so that the first 32 entries of the array are the red channel 
#			values of the first row of the image.
# labels -- a list of 10000 numbers in the range 0-9. The number at index
# 			i indicates the label of the ith image in the array data.
# keys: [b'batch_label', b'labels', b'data', b'filenames']
batch1 = unpickle('CIFAR10_batches/data_batch_1')
batch2 = unpickle('CIFAR10_batches/data_batch_2')
batch3 = unpickle('CIFAR10_batches/data_batch_3')
batch4 = unpickle('CIFAR10_batches/data_batch_4')
batch5 = unpickle('CIFAR10_batches/data_batch_5')

# For 'CIFAR10_batches/test_batch':
batch_test = unpickle('CIFAR10_batches/test_batch')

#########################################################
#--------- funcation: display_picture_CIFAR -------------
# Description: 	Display a picture from CIFAR
# Input      :	batch:	the batch of target picture
# 				index:	the number of target picture
# Output     :	None
def display_picture_CIFAR(batch, index):
	picture_data = batch[b'data'][index]
	picture_label = batch[b'labels'][index]
	picture = picture_data.reshape(3,32,32)
	picture = picture.transpose(1, 2, 0)		# Exchange of dimension
	img = Image.fromarray(picture,'RGB')		# numpy -> image
	plt.imshow(img)
	label = batches_meta[b'label_names'][picture_label] # label translation
	label = label.decode('ascii')			#decode so invert to string
	plt.title(label)
	plt.show()

#----------- funcation: KNN_recongnition ---------------
# Description: 	discern a unknown picture by KNN method
# Input      :	batch:	 		the batch of target picture
# 				index:	 		the number of target picture
#				train_quantity: quantity of train data
#				k:		 		use the nearest k label
# Output     :	judge:	 		True is success, False is error
#				label_test: 	label by using KNN
#				label_real: 	actual label
# TEST result:  k = 15:  13.7%
#				k = 7 :   9.5%
#				k = 25:  14.3%  
#				k = 30:  17.0%  
def KNN_recongnition( batch , index , train_quantity = 10000 , k = 30 ):
	unknown_data = batch[b'data'][index]
	dist_list = np.zeros((k, 2))

	if train_quantity <= 2*k:
		print('train_quantity is too few!')
		exit(1)

	for ii in range(0, k):
		contrast_data = batch1[b'data'][ii]
		dist_list[ii, 0] = np.linalg.norm(unknown_data - contrast_data)	# Euclidean distance
		dist_list[ii, 1] = batch1[b'labels'][ii]

	if train_quantity <= 10000:
		dist_list = pop_new_data(batch1, dist_list, unknown_data, k, train_quantity)	
	elif train_quantity <= 20000:
		dist_list = pop_new_data(batch1, dist_list, unknown_data, k, 10000)
		dist_list = pop_new_data(batch2, dist_list, unknown_data, 0, train_quantity % 10000)	
	elif train_quantity <= 30000:	
		dist_list = pop_new_data(batch1, dist_list, unknown_data, k, 10000)	
		dist_list = pop_new_data(batch2, dist_list, unknown_data, 0, 10000)
		dist_list = pop_new_data(batch3, dist_list, unknown_data, 0, train_quantity % 10000)
	elif train_quantity <= 40000:	
		dist_list = pop_new_data(batch1, dist_list, unknown_data, k, 10000)	
		dist_list = pop_new_data(batch2, dist_list, unknown_data, 0, 10000)
		dist_list = pop_new_data(batch3, dist_list, unknown_data, 0, 10000)
		dist_list = pop_new_data(batch4, dist_list, unknown_data, 0, train_quantity % 10000)
	elif train_quantity <= 50000:	
		dist_list = pop_new_data(batch1, dist_list, unknown_data, k, 10000)	
		dist_list = pop_new_data(batch2, dist_list, unknown_data, 0, 10000)	
		dist_list = pop_new_data(batch3, dist_list, unknown_data, 0, 10000)
		dist_list = pop_new_data(batch4, dist_list, unknown_data, 0, 10000)
		dist_list = pop_new_data(batch5, dist_list, unknown_data, 0, train_quantity % 10000)	
	else:
		print('train_quantity is too many!')
		exit(1)

	cnt_list = np.zeros((10,1))		# count the output category
	print(dist_list.T)

	for ii in dist_list[:,1]:
		cnt_list[int(ii)] += 1

	category = np.argmax(cnt_list)

	if category == batch[b'labels'][index]:
		label = batches_meta[b'label_names'][category].decode('ascii')
		return (True, label, label)
	else:
		label_test = batches_meta[b'label_names'][category].decode('ascii')
		label_real = batches_meta[b'label_names'][batch[b'labels'][index]].decode('ascii')
		return (False, label_test, label_real)

#--------------- funcation: pop_new_data ------------------
def pop_new_data(batch, dist_list, unknown_data, train_start, train_end ):
	for ii in range(train_start, train_end):
		contrast_data = batch[b'data'][ii]
		temp_dist = np.linalg.norm(unknown_data - contrast_data)	# Euclidean distance
		max_dist_in_list = max(dist_list[:, 0])

		if max_dist_in_list > temp_dist:
			arg_dist = np.argmax(dist_list[:, 0])
			dist_list[arg_dist, 0] = temp_dist
			dist_list[arg_dist, 1] = batch[b'labels'][ii]
	return dist_list

# quantity of train data is 5000, and quantity of test data is 500
train_num = 50000
test_num = 2000

log_file = open('log.txt', 'a+')	# file can be write or add.
for kk in [10]:				# kk represent the k of KNN Algorithm, CS231n’s n is 10.
	cnt_true = 0					# cnt_true represent the present true count pre epoch 
	cnt_test = 0
	# Randomly extract data in batch_test 
	for ii in range(0,test_num): 	# [random.randint(0,10000 - 1) for _ in range(test_num)]:
		# use KNN Algorithm predict the lablel of unknow input data.
		(judge, label_test, label_real) = KNN_recongnition(batch_test, ii, train_num, kk)
		cnt_test += 1
		if judge:
			cnt_true += 1
		print(' cnt_test: ', cnt_test,' cnt_true: ', cnt_true,' Accuracy rate: ', cnt_true/float(cnt_test))
		# display_picture_CIFAR(batch_test, ii)
	print('k = ', kk, 'Accuracy rate = ', cnt_true/float(test_num),'+++++++++++++++++++++++++++++++')
	log_file.write('train:%d\ttest:%d\tk:%d\tAcc:%f\n'%(train_num, test_num, kk, cnt_true/float(test_num)))

print('--------------------------------------------------------------------')
log_file.close()

