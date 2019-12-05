#############################################
# author : Baptiste PICARD 		 			#
# date : 01/09/2019				 			#
# for : UNcuyo PFE 							#
# last change : -							#
# review : 									#
#											# 
# overview : This script gives special 		#
# functions to help.						#
#############################################

# All the imports
# My files .py
import navigate 
# Librairies
import sys
import os
import json
import ast # Read json files.
import shutil # Remove a dir with subdirs.
import subprocess # Change security.
from datetime import datetime
import matplotlib.pyplot as plt # Draw figures
from PyQt5.QtWidgets import QApplication # get DPI of my screen
import pandas as pd # Librairy to read a csv.

# Functions validated.
def createJSON(project_path, json_file):
	data_for_json = {
		"project_path" : str(project_path),
		"datasets_path" : str(project_path)+"\\Datasets",
		"dataset_mp3_sorted_path" : str(project_path)+"\\Datasets"+"\\Mp3Sorted",
		"dataset_mp3_path" : str(project_path)+"\\Datasets"+"\\Mp3",
		"dataset_wav_path" : str(project_path)+"\\Datasets"+"\\Wav",
		"dataset_image_path" : str(project_path)+"\\Datasets"+"\\Images",
		"dataset_wav_gtzan_path" : str(project_path)+"\\Datasets"+"\\WavGTZAN",
		"dataset_images_gtzan_path" : str(project_path)+"\\Datasets"+"\\ImagesGTZAN",
		"results_folder" : str(project_path)+"\\Results",
		"list_of_items_path" : ["Datasets", "Results", "config.json", "main.py", "model.py", "navigate.py", "songs.py", "threads.py", "utils.py", "tracks.csv", "train_model.ipynb", "librosa_test.ipynb"],
		"list_of_items_dataset" : ["Images", "Mp3", "Mp3Sorted", "Wav", "WavGTZAN", "ImagesGTZAN"],
		"scores_file" : "classification_results.txt"
	}
	if not(os.path.exists(json_file)) :
		print("Creating the .JSON file.")
		open(json_file, "w+")
		with open(json_file, "w") as write_file:
			json.dump(data_for_json, write_file)

def openJSON(json_file) :
	if(os.path.exists(json_file)) :
		print("Opening the .JSON file : ",json_file," !")
		with open(json_file, "r") as read_file:
			return ast.literal_eval(read_file.read())

def check(my_json) :
	print("Checking the project.")
	navigate.set_path_to_project_path(my_json) 
	for sub in get_list_subs() :
		if(sub!="__pycache__" and sub!=".ipynb_checkpoints" and sub!="train_model.ipynb"):	
			if(sub in my_json["list_of_items_path"]) : 
				pass
			else : 
				print(sub," isn't existing yet, there is a problem.\n")
	navigate.set_path_to_datasets(my_json) 
	for sub in get_list_subs() :	
		if(sub in my_json["list_of_items_dataset"]) : 
			pass
		else : 
			print(sub," isn't existing yet, there is a problem.\n")

def razJSON() : 
	print("Deleting the .JSON file.")
	if(os.path.exists("config.json")) :
		os.remove('config.json')

def razImages(my_json) : 
	print("Erasing all the images.")
	navigate.set_path_to_dataset_image(my_json)
	for x_dir in get_list_subs() :
		shutil.rmtree(x_dir)
	navigate.set_path_to_project_path(my_json)

def razWAV(my_json) : 
	print("Erasing all the wav files.")
	navigate.set_path_to_dataset_WAV(my_json)
	for x_dir in get_list_subs() :
		shutil.rmtree(x_dir)
	navigate.set_path_to_project_path(my_json)	

def razMP3Sorted(my_json) : 
	print("Erasing all the mp3 songs in MP3Sorted directory.")
	navigate.set_path_to_dataset_MP3_sorted(my_json)
	for x_dir in get_list_subs() :
		shutil.rmtree(x_dir)
	navigate.set_path_to_project_path(my_json)

def razImagesGTZAN(my_json) : 
	print("Erasing all the mp3 songs in MP3Sorted directory.")
	navigate.set_path_to_dataset_image_GTZAN(my_json)
	for x_dir in get_list_subs() :
		shutil.rmtree(x_dir)
	navigate.set_path_to_project_path(my_json)

def raz(my_json) :
	razMP3Sorted(my_json)
	razWAV(my_json)
	razImages(my_json)

def transformNameDirectory(actual_name) :
	impossible_chars = ['*', '?', ':', '/', '>', '<', '|', '"', ' ']
	for c in impossible_chars :
		actual_name = actual_name.replace(c, '_')
	return actual_name

def getNumberOfTracks(my_json) :
	total = 0
	navigate.set_path_to_dataset_WAV(my_json)
	for x_dir in navigate.get_list_subs() :
		l_dir = len(os.listdir(navigate.get_actual_path()+"\\"+x_dir))
		total += l_dir
		print("We have ",l_dir," songs in the ",x_dir," directory.")
	print("In total, there is ",total," tracks.")


def get_list_subs() :
	"""
		overview :  Get a list of subs from the actual path.
		output :
			- subs : list which contains all the items of the actual path.
	"""
	subs = [dir_name for dir_name in os.listdir(navigate.get_actual_path())]
	return subs

def set_security_autorizations_images(my_json) :
	print("Setting the autorization of the image folders.")
	navigate.set_path_to_dataset_image(my_json)
	for dir_x in get_list_subs() :
		navigate.set_path(dir_x)
		# print("Path : ",navigate.get_actual_path(),".\n")
		os.chmod(navigate.get_actual_path(), 0o777)
		navigate.set_path_to_dataset_image(my_json)

def get_DPI() : 
	app = QApplication(sys.argv)
	screen = app.screens()[0]
	dpi = screen.physicalDotsPerInch()
	app.quit()
	return dpi


def saveResults(my_json, model, X_train, X_test, Y_train, scores, epochs, batch_size, my_time, nfft, overlap, Fs, optimizer, loss_func, frequency, categories, encoder, id_model) :
	navigate.set_path_to_results(my_json)
	if not(os.path.exists(my_json['scores_file'])) :
		open(my_json['scores_file'], 'a').close()
		print("File : ",my_json['scores_file']," created.")
	else :
		print("Writing the results (information) of the training dataset in : ",my_json['scores_file'],".")
		f = open(my_json['scores_file'],"a")
		f.write(str("---- Results of "+str(datetime.now())+" ----\n"))
		f.write(str("Categories : "+str(categories)+" Hz.\n"))
		if(id_model==0) :
			f.write(str("Original frequency of a song : "+str(frequency)+" Hz.\n"))
			f.write(str("Frequency of a song : "+str(Fs)+" Hz.\n"))
			f.write(str("NFFT of a song : "+str(nfft)+".\n"))
			f.write(str("overlap of a song : "+str(overlap)+".\n"))
		else :
			pass
		f.write(str("Optimizer of the model : "+str(optimizer)+" .\n"))
		f.write(str("Loss function of the model : "+str(loss_func)+" .\n"))
		f.write(str("Totat dataset : "+str(len(X_train)+len(X_test))+" items.\n"))
		f.write(str("Length train set : "+str(len(X_train))+".\n"))
		f.write(str("Length test set : "+str(len(X_test))+".\n"))
		f.write(str("Epochs : "+str(epochs)+".\n"))
		f.write(str("Batch Size : "+str(batch_size)+".\n"))
		f.write(str("Loss : "+str(int(scores[0]*100))+"%.\n"))
		f.write(str("Accuracy : "+str(int(scores[1]*100))+"%.\n"))
		f.write(str("Using encoder to encode outputs : "+str(encoder)+".\n"))
		f.write(str("It takes : "+str(int(my_time/60))+" minutes to train my model.\n"))
		model.summary(print_fn=lambda x: f.write(x + '\n'))
		f.write(str("------------------------------------------------------------------\n\n"))
		f.close()
	navigate.set_path_to_project_path(my_json)

def printLossAccu(my_json, model_fitted) : 
	print("Saving loss and accuracy figures.\n")
	# summarize history for accuracy
	fig_acc = plt.figure()
	plt.plot(model_fitted.history['acc'])
	plt.plot(model_fitted.history['val_acc'])
	plt.title(str('Model accuracy : '+str(datetime.now())))
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	actual_date = str(datetime.now())
	actual_date = actual_date.replace(" ", "_")
	actual_date = actual_date.replace(":", "_")
	actual_date = actual_date.replace("-", "_")
	fig_acc.savefig(str(my_json["results_folder"]+"\\model_accuracy_"+actual_date+".png"))
	plt.close()
	# summarize history for loss
	fig_loss = plt.figure()
	plt.plot(model_fitted.history['loss'])
	plt.plot(model_fitted.history['val_loss'])
	plt.title(str('Model loss : '+str(datetime.now())))
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	fig_loss.savefig(str(my_json["results_folder"]+"\\model_loss_"+actual_date+".png"))
	plt.close()
	