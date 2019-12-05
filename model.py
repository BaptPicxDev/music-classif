#############################################
# author : Baptiste PICARD 		 			#
# date : 06/10/2019				 			#
# for : UNcuyo PFE 							#
# last change : -							#
# review : 									#
# 								 			#
# overview : This script provide a neural   #
# Network model to give the ability to my 	#
# application learn about songs and 		#
# classify them into genders.
#############################################

# All the imports
# My scripts
import utils
import navigate
# Librairies
import random
import ast
from PIL import Image
import numpy as np
import librosa as lr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Activation, Input, Flatten

def shapeDataConv2D(my_json, percent_train, encoder=False) :
	"""
		overview :  Shape the datas to return the train and test sets.
		input :
			- my_json : .json file with all the datas we need.
			- percent_train : float -> [0.0 : 1.0] which represent the percentage 
			of images which will be include in the train set.
	"""
	print("Shaping the data to get the Train and Test sets.")
	navigate.set_path_to_dataset_image(my_json)
	X, Y = [], []
	path = navigate.get_actual_path()
	maxi = len(utils.get_list_subs())
	for index_dir, dir_x in enumerate(utils.get_list_subs()) :
		navigate.set_path(dir_x)
		for index_fig, fig in enumerate(utils.get_list_subs()) :
			im = Image.open(navigate.get_actual_path()+"\\"+fig).convert('L') # 640 * 480 pixels
			width, height = im.size
			pix = np.array(im.getdata()).reshape(width, height)
			if(encoder) :
				category = str(im.info["category"])
			else :	
				category = ast.literal_eval(im.info["category_shaped"])
			im.close()
			X.append(pix)
			Y.append(category)
		navigate.set_path_to_dataset_image(my_json)
	if(encoder) :
		Y = encoder.fit_transform(Y)
		n_classes = encoder.classes_.size
		Y = Y.reshape(len(Y), 1)
	else :
		Y = reshapeOutputs(Y)
	X = reshapeEntriesConv2D(X, width, height)	

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(1 - percent_train))
	return X_train, Y_train, X_test, Y_test, width, height

def shapeDataConv2DGTZAN(my_json, percent_train) :
	"""
		overview :  Shape the datas to return the train and test sets.
		input :
			- my_json : .json file with all the datas we need.
			- percent_train : float -> [0.0 : 1.0] which represent the percentage 
			of images which will be include in the train set.
	"""
	print("Shaping the data to get the Train and Test sets.")
	navigate.set_path_to_dataset_image_GTZAN(my_json)
	X, Y = [], []
	path = navigate.get_actual_path()
	maxi = len(utils.get_list_subs())
	convert_str = "L" # L=B&W, RGB
	for index_dir, dir_x in enumerate(utils.get_list_subs()) :
		navigate.set_path(dir_x)
		for index_fig, fig in enumerate(utils.get_list_subs()) :
			im = Image.open(navigate.get_actual_path()+"\\"+fig).convert(convert_str) # 640 * 480 pixels
			width, height = im.size
			pix = np.array(im.getdata())
			im.close()
			X.append(pix)
			Y.append(dir_x)
		navigate.set_path_to_dataset_image_GTZAN(my_json)
	if(convert_str == "L") :
		X = reshapeEntriesConv2D(X, width, height, 1)	
	if(convert_str == "RGB") :
		X = reshapeEntriesConv2D(X, width, height, 3)	
	Y = reshapeOutputsDense(Y)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(1 - percent_train))
	return X_train, Y_train, X_test, Y_test, width, height

def shapeDataDense(my_json, percent_train, encoder=False) :
	"""
		overview :  Shape the datas to return the train and test sets.
		input :
			- my_json : .json file with all the datas we need.
			- percent_train : float -> [0.0 : 1.0] which represent the percentage 
			of images which will be include in the train set.
	"""
	print("Shaping the data to get the Train and Test sets.")
	navigate.set_path_to_dataset_WAV_GTZAN(my_json)
	X, Y = [], []
	path = navigate.get_actual_path()
	maxi = len(utils.get_list_subs())
	encoder = LabelEncoder()
	for index_dir, dir_x in enumerate(utils.get_list_subs()) :
		navigate.set_path(dir_x)
		for index_fig, audio_path in enumerate(utils.get_list_subs()) :
			print("Index folder : ",index_dir+1, "/10 -- File ",index_fig+1,"/100.")
			audio, freq = lr.load(audio_path, mono=True)
			chroma_stft = lr.feature.chroma_stft(y=audio, sr=freq)
			rmse = lr.feature.rmse(y=audio)
			spec_cent = lr.feature.spectral_centroid(y=audio, sr=freq)
			spec_bw = lr.feature.spectral_bandwidth(y=audio, sr=freq)
			rolloff = lr.feature.spectral_rolloff(y=audio, sr=freq)
			zcr = lr.feature.zero_crossing_rate(audio)
			mfcc = lr.feature.mfcc(y=audio, sr=freq)
			data = [np.mean(chroma_stft), np.mean(rmse), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)]
			for mfcc_x in mfcc : 
				data.append(np.mean(mfcc_x))
			X.append(data)
			Y.append(dir_x)
			break
		navigate.set_path_to_dataset_WAV_GTZAN(my_json)
	X = reshapeEntriesDense(X)	
	Y = reshapeOutputsDense(Y)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(1 - percent_train))
	return X_train, Y_train, X_test, Y_test

def reshapeEntriesConv2D(X, width, height, num) :
	print("Reshapping inputs.")
	X = np.array(X)
	X = X.reshape(len(X), width, height, num)
	X = X.astype('float32')/255
	return X

def reshapeEntriesDense(X) :
	print("Reshapping inputs.")
	X = np.array(X, dtype='float')
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	X = X.reshape(len(X), len(X[0]))
	return X

def reshapeOutputsConv2D(Y) :
	print("Reshapping outputs.")
	return np.array(Y)


def reshapeOutputsDense(Y) :
	print("Reshapping outputs.")	
	new_Y = Y
	encoder = LabelEncoder()
	Y_fitted = encoder.fit_transform(Y)
	n_classes = len(encoder.classes_)
	for i in range(len(Y)) :
		new_Y_i=np.zeros(n_classes)
		new_Y_i[Y_fitted[i]] = 1
		new_Y[i] = new_Y_i
	return np.array(new_Y)

def shapeOutputs(my_json) :
	navigate.set_path_to_dataset_WAV(my_json)
	n_cat = len(utils.get_list_subs())
	categories = []
	for index, item in enumerate(utils.get_list_subs()) :
		my_array = [0] * n_cat
		my_array[index] = 1
		categories.append({'category_name' : item , 'category_array' : my_array})
	return categories

def getConv2DModel(n_classes, width_image, height_image, optimizer, loss_func): 
	print("Getting the Convolution 2 dimensions model !")
	model = Sequential()
	model.add(Conv2D(filters=64, kernel_size=(2,2), input_shape=(width_image, height_image, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(filters=128, kernel_size=(2,2), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(filters=256, kernel_size=(2,2), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(filters=512, kernel_size=(2,2), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	# model.add(Dense(1024, activation='relu'))
	model.add(Dense(128, activation='relu'))
	# model.add(Dropout(0.5))
	model.add(Dense(n_classes, activation="softmax")) # 2 - 3 capas Dense
	model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy', f1_m, precision_m, recall_m]) # Adam ?
	return model

def getConv2DModel1(n_classes, width_image, height_image, optimizer, loss_func): 
	print("Getting the Convolution 2 dimensions model !")
	model = Sequential()
	model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(width_image, height_image, 1), activation='relu'))
	model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(n_classes, activation="softmax")) # 2 - 3 capas Dense
	model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy', f1_m, precision_m, recall_m]) # Adam ?
	return model

def getDenseModel(X, num_classes, optimizer, loss_func): 
	print("Getting the Dense model !")
	model = Sequential()
	model.add(Dense(512, input_shape=(X.size,), activation='relu'))
	model.add(Dense(256, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(64, activation='relu'))
	# model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation="softmax")) # 2 - 3 capas Dense
	model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]) # Adam ?
	return model

def fitModel(model, X_train, Y_train, X_test, Y_test, epochs, batch_size, verbose, history) : 
	model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[history])
	return model

