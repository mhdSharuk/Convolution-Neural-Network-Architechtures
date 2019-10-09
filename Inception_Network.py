#Importing the required modules

import tensorflow as tf 
from tensorflow.keras import Model,Input
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten
from tensorflow.keras.layers import Dense,Concatenate,Activation

def inception_network_v1(input_shape, num_classes):

	"""Arguments:
		input_shape : The number of rows,number of columns,number of color channels in the image
					  Example : (512,512,3) => This represents 512x512 pixel image with 3 color channels
		num_classes : The number of classes we want to predict

	   Output:
	   	Return the Inception model with the given constraints
	"""

	input_ = Input(input_shape)

	#==============================================================================================================================
	#Strating of First BLock
	tower1_1 = Conv2D(filters = 64, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_)

	tower2_1 = Conv2D(filters = 64, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_)
	tower2_2 = Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu')(tower2_1)

	tower3_1 = Conv2D(filters = 64, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_)
	tower3_2 = Conv2D(filters = 64, kernel_size = (5,5), padding = 'same', activation = 'relu')(tower3_1)

	tower4_1 = MaxPooling2D(pool_size = (3,3), strides = (1,1), padding = 'same', activation='relu')(input_)
	tower4_2 = Conv2D(filters = 64, kernel_size = (1,1), padding = 'same', activation = 'relu')(tower4_1)

	concat_1 = Concatenate([tower1_1,tower2_1,tower3_1,tower4_1], axis = 3)
	#First BLock Finished

	#===============================================================================================================================
	#Same for Every end of the block with flatten function refering to the last concatenation layer
	end_block = Conv2D(filters = 8, kernel_size = (3,3))(concat_1)
	end_block = Activation('relu')(end_block)
	end_block = MaxPooling2D(pool_size = (2,2), strides = (1,1))(end_block)
	end_block = Flatten()(end_block)
	end_block = Dense(num_classes)(end_block)
	
	output = Activation('softmax')(end_block)

	model = Model([input_], output)

	return model

