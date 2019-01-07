'''-------------------------------------------------------
                          \\\||///
                         { -    - }
	           [][][_]  [  @    @  ]  [_][][]
----------------------------------------------------------
Author		:		Yopen Hu
foldername	:		CIFAR10_train
filename	:   	Data_Preprocess.py
Description	:		
Date  	  By   Version        Change Description
=========================================================
18/11/13  HYP    0.1       extract data & read data
---------------------------------------------------------
                                  0ooo
------------------------ooo0-----(    )------------------
                       (    )     )  /
                        \  (     (__/
                         \__)
------------------------------------------------------'''
import numpy as np 
import matplotlib.pyplot as plt 	# plot 2D data
from PIL import Image				# Python Image Library

#------------------- funcation: unpickle ----------------
# Description: 	Offical data package decode
# Input      :	file:	the location of package
# Output     :	dict:	the data sheet from file
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

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
	label = batches_meta[b'label_names'][picture_label]		# label translation
	label = label.decode('ascii')				#decode so invert to string
	plt.title(label)
	plt.show()

################################### load data #############################
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
# batch2 = unpickle('CIFAR10_batches/data_batch_2')
# batch3 = unpickle('CIFAR10_batches/data_batch_3')
# batch4 = unpickle('CIFAR10_batches/data_batch_4')
# batch5 = unpickle('CIFAR10_batches/data_batch_5')



display_picture_CIFAR(batch1, 1000)

