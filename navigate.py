#############################################
# author : Baptiste PICARD 		 			#
# date : 21/08/2019				 			#
# for : UNcuyo PFE 							#
# last change : 26/08/2018					#
# review : Creating functions to help 		#
# navigating and i synthesized some 		#
# functions.								#
# 								 			#
# overview : This script exists to give		#
# the different functions to help naviga -	#
# ting between the different folders etc .. #
#############################################

# All the imports.
import os # Help to naviguate into a folder.

def get_actual_path() :
	"""
		overview :  Get the actual path
		output :
			- actual_path : string which represents the actual path.
	"""
	return os.getcwd()

def set_path_to_project_path(my_json) :
	"""
		overview :  Set the actual path to project path
		input :
			- project_path : string which represents the project path.
	"""
	os.chdir(my_json["project_path"]) 


def set_path_to_datasets(my_json) :
	"""
		overview :  Set the actual path to project path
		input :
			- project_path : string which represents the project path.
	"""
	os.chdir(my_json["datasets_path"]) 


def set_path_to_dataset_MP3(my_json) : 
	"""
		overview :  Set the actual path to dataset path
		input :
			- project_path : string which represents the project path.
	"""
	os.chdir(my_json["dataset_mp3_path"])


def set_path_to_dataset_MP3_sorted(my_json) : 
	"""
		overview :  Set the actual path to dataset path
		input :
			- project_path : string which represents the project path.
	"""
	os.chdir(my_json["dataset_mp3_sorted_path"])

def set_path_to_dataset_WAV(my_json) : 
	"""
		overview :  Set the actual path to dataset path
		input :
			- project_path : string which represents the project path.
	"""
	os.chdir(my_json["dataset_wav_path"])

def set_path_to_dataset_image(my_json) : 
	"""
		overview :  Set the actual path to dataset path
		input :
			- project_path : string which represents the project path.
	"""
	os.chdir(my_json["dataset_image_path"])

def set_path_to_dataset_WAV_GTZAN(my_json) : 
	"""
		overview :  Set the actual path to dataset path
		input :
			- project_path : string which represents the project path.
	"""
	os.chdir(my_json["dataset_wav_gtzan_path"])

def set_path_to_dataset_image_GTZAN(my_json) : 
	"""
		overview :  Set the actual path to dataset path
		input :
			- project_path : string which represents the project path.
	"""
	os.chdir(my_json["dataset_images_gtzan_path"])

def set_path_to_results(my_json) : 
	"""
		overview :  Set the actual path to dataset path
		input :
			- project_path : string which represents the project path.
	"""
	os.chdir(my_json["results_folder"])

def set_path(path) :
	"""
		overview :  Set the actual path to a new path, using a diretory in the actual directory.
		intput :
			- new_directory : string which represents the directory where i want to go.
	"""
	os.chdir(path)

