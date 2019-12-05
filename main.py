#############################################
# author : Baptiste PICARD 		 			#
# date : 21/08/2019				 			#
# for : UNcuyo PFE 							#
# last change : 31/08/2019					#
# review : Modifications of the script,		#
# synthetized the code and made subs .py 	#
# files.									#
# 								 			#
# overview : This script is the main script #
# for the project of music classification 	#
# into gender.								# 
#############################################

# All the imports 
# Librairies
import os # Help to get the path.
import time # This librairy permit to calculate the duration of the main script.
import shutil # Remove a dir with subdirs.
import pandas as pd 
import multiprocessing as mp
import matplotlib as mlp
mlp.use("Agg") # Set the environnement to avoid displaying figures.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Avoid FutureWarning -> h5py from 'float' to 'np.float64'
# My scripts 
import songs # A python file i created.
import navigate # A python file i created.
import utils # Utilities funcions.
import model
from threads import WavThread, ImageProcess, SortMP3Thread, ImageProcessGTZAN # Threads to accelerate the process.

# Variables 
if(os.path.exists("src")) :
	project_path = str(navigate.get_actual_path()+"\\src") # This is the path to the project folder.
else : 
	project_path = navigate.get_actual_path() # This is the path to the project folder.
json_file = "config.json"
tracks = "tracks.csv"

# Functions

def initialiseMP3Sorted(my_json) : # -> 1748 secs, no probs
	print("Sorting the MP3 files.")
	datas = pd.read_csv(tracks, encoding="utf-8", usecols=["album", 'track.19', "track.7", "track.16"], dtype = {"album" : object, 'track.19' : object, "track.7" : object, "track.16" : object})
	datas = datas.rename(columns={"album" : "track_id", 'track.19' : "title", "track.7" : "genre_top", "track.16" : "number"})
	threads = []
	navigate.set_path_to_datasets(my_json)
	if not(os.path.exists("Mp3Sorted")) :
		os.makedirs("Mp3Sorted")
	navigate.set_path_to_dataset_MP3(my_json)
	for index_dir, x_dir in enumerate(utils.get_list_subs()) :
		if(x_dir != 'checksums' and x_dir!="fma_metadata" and x_dir!="README.txt") :
			navigate.set_path(x_dir)
			actual_dataset_path = navigate.get_actual_path()
			items_dataset = navigate.get_list_subs() # Return all the track of the actual folder.
			my_thread = SortMP3Thread("MP3 Sorted Thread "+str(index_dir), my_json, actual_dataset_path, x_dir, items_dataset, datas)
			threads.append(my_thread)
		else : 
			print("This is not a directory of mp3 tracks.")
		navigate.set_path_to_dataset_MP3(my_json)
	navigate.set_path_to_project_path(my_json)
	for thread in threads : 
		thread.start()
	for thread in threads : 
		thread.join()
	navigate.set_path_to_project_path(my_json)

def initialiseWAV(my_json): # -> 1716 secs, no probs
	print("Creating the WAV files.")
	navigate.set_path_to_dataset_MP3_sorted(my_json)
	all_datasets = utils.get_list_subs() # List all the sub folders of the dataset folder.
	threads = []
	for index_d, dataset in enumerate(all_datasets) : 
		if(len(os.listdir(my_json['dataset_mp3_sorted_path']+'\\'+dataset)) >= 30) : 
			navigate.set_path(my_json["dataset_mp3_sorted_path"]+"\\"+dataset) # Set the path to the first sub dataset directory, which contains tracks. 
			actual_dataset_path = navigate.get_actual_path()
			items_dataset = navigate.get_list_subs() # Return all the track of the actual folder.
			my_thread = WavThread("Wav Thread "+str(index_d), my_json, actual_dataset_path, dataset, items_dataset)
			threads.append(my_thread)
			navigate.set_path_to_dataset_MP3_sorted(my_json)
		else : 
			print("There is less than 30 songs in the ",dataset," dataset. I will not export it.")
	navigate.set_path_to_project_path(my_json)
	for thread in threads : 
		thread.start()
	for thread in threads : 
		thread.join()
	navigate.set_path_to_project_path(my_json)

def initialiseFigs(my_json, dpi, height, width, legend, colormesh) : # 1444 secs
	# We can't use threat, because matplotlib isn't thread protected.
	print("Creating the Images.")
	navigate.set_path_to_dataset_WAV(my_json)
	all_datasets = utils.get_list_subs() # List all the sub folders of the dataset folder.
	navigate.set_path_to_dataset_image(my_json)
	for dataset in all_datasets : 
		if not(os.path.exists(dataset)) :
			os.makedirs(dataset)
	navigate.set_path_to_dataset_WAV(my_json)
	processes = []
	outputs_shaped = model.shapeOutputs(my_json)
	right_array = 0
	flag = False
	for index_d, dataset in enumerate(all_datasets) :
		for item in outputs_shaped : 
			if(item['category_name'] == dataset) :
				right_array = item["category_array"]
				flag = True
		navigate.set_path(dataset) # Set the path to the first sub dataset directory, which contains tracks. 
		actual_dataset_path = navigate.get_actual_path()
		items_dataset = utils.get_list_subs() # Return all the track of the actual folder.
		if(len(items_dataset)>= 50 and flag==True) : # 
			my_process = ImageProcess("Image process "+str(index_d), my_json, actual_dataset_path, dataset, items_dataset, right_array, my_json["dataset_image_path"]+"\\"+dataset, dpi, height, width, legend=legend, colormesh=colormesh)
			processes.append(my_process)
		navigate.set_path_to_dataset_WAV(my_json)
	processes = [mp.Process(target=p.run, args=()) for p in processes]
	for p in processes :
		p.start()
	for p in processes :
		p.join()
	navigate.set_path_to_project_path(my_json)

def initialiseFigsGTZAN(my_json, dpi, height, width, legend) : # 1444 secs
	# We can't use threat, because matplotlib isn't thread protected.
	print("Creating the Images.")
	navigate.set_path_to_dataset_WAV_GTZAN(my_json)
	all_datasets = utils.get_list_subs() # List all the sub folders of the dataset folder.
	navigate.set_path_to_dataset_image_GTZAN(my_json)
	for dataset in all_datasets : 
		if not(os.path.exists(dataset)) :
			os.makedirs(dataset)
	navigate.set_path_to_dataset_WAV_GTZAN(my_json)
	processes = []
	for index_d, dataset in enumerate(all_datasets) :
		navigate.set_path(dataset) # Set the path to the first sub dataset directory, which contains tracks. 
		actual_dataset_path = navigate.get_actual_path()
		items_dataset = utils.get_list_subs() # Return all the track of the actual folder.
		if(len(items_dataset)>= 50) : # 
			my_process = ImageProcessGTZAN("Image GTZAN process "+str(index_d), my_json, actual_dataset_path, dataset, items_dataset, my_json["dataset_images_gtzan_path"]+"\\"+dataset, dpi, height, width, legend=legend)
			processes.append(my_process)
		navigate.set_path_to_dataset_WAV_GTZAN(my_json)
	processes = [mp.Process(target=p.run, args=()) for p in processes]
	for p in processes :
		p.start()
	for p in processes :
		p.join()
	navigate.set_path_to_project_path(my_json)


# Principal script.
if __name__ == "__main__" :
	print("This is the beginning of the script which will classify songs into genders.")
	start = time.clock() # Start 

	# utils.razJSON()
	# utils.createJSON(project_path, json_file)
	my_json = utils.openJSON(json_file)
	screen_dpi = utils.get_DPI()
	# utils.razImages(my_json)
	# utils.razImagesGTZAN(my_json)
	# utils.check(my_json)
	# utils.raz(my_json)
	# initialiseMP3Sorted(my_json)
	# initialiseWAV(my_json) # -> 30 mins.
	# initialiseFigs(my_json, dpi=screen_dpi, height=128, width=128, legend=False, colormesh=False) # -> 30 mins + bugs.
	# initialiseFigsGTZAN(my_json, screen_dpi, 480, 640, legend=False)
	# songs.getSongsFigures(my_json, 'D:\\Projets\\music-classif\\volumes\\Datasets\\Wav\\Electronic\\003573.wav', True)
	songs.getSpectrogram(my_json, 'D:\\Projets\\music-classif\\volumes\\Datasets\\WavGTZAN\\metal\\metal.00012.wav', True)
	end = time.clock()-start # End
	print("It takes ",end," seconds to reach the end of the program.")