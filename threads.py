
# All the imports
# My Scripts 
import navigate
import utils
import songs
# Librairies
import os 
import scipy
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import librosa as lr
import librosa.display
from pydub import * # Convert mp3 files
from multiprocessing import Process
from threading import Thread, RLock
from PIL import Image, PngImagePlugin


class WavThread(Thread) :
	"""
		overview : Thread which permite filling the database.
		We have 2k items, and it takes an average of 5 seconds per item.
		We will use Threads to make it faster.
	"""
	def __init__(self, name, my_json, path, category, tracksmp3) :
		Thread.__init__(self)
		print("Creating the Thread : ",name)
		self.name = name
		self.my_json = my_json
		self.old_path = navigate.get_actual_path()
		self.path = path
		self.category = category
		self.tracksmp3 = tracksmp3

	def run(self) :
		print("Running the Thread : ",self.name)
		for i, item in enumerate(self.tracksmp3) :
			if(item[-3:]=="mp3") : # Need to see if there is already a song_name.wav.
				if not os.path.exists(self.my_json["dataset_wav_path"]+"\\"+self.category):
					os.makedirs(self.my_json["dataset_wav_path"]+"\\"+self.category)
					print("Directory ",self.category,"created.")
				mp3_song = AudioSegment.from_mp3(self.my_json["dataset_mp3_sorted_path"]+"\\"+self.category+"\\"+item)
				if not os.path.exists(self.my_json['dataset_wav_path']+"\\"+self.category+'\\'+item.replace(".mp3",".wav")) :
					mp3_song = mp3_song.set_channels(1)
					mp3_song.export(self.my_json['dataset_wav_path']+"\\"+self.category+'\\'+item.replace(".mp3",".wav"), format="wav")
					print(self.name," Index ",i,". The song : ",item, " was export to .wav.")
				else : 
					print(self.name," Index ",i,". The song : ",item.replace('mp3','wav'), " already exists.")
			else : 
				print(self.name," : Error exporting Mp3 -> Wav. Not the good format.")
		print("The thread : ",self.name, " is terminated.")

class ImageProcess() :
	"""
		overview : Thread which permite filling the database.
		We have 2k items, and it takes an average of 5 seconds per item.
		We will use Threads to make it faster.
	"""
	def __init__(self, name, my_json, path, category, trackswav, corresponding_cat_array, dest, dpi, width, height, legend=False, colormesh=False) :
		print("Creating the Multiprocess class ",name)
		self.name = name
		self.my_json = my_json
		self.old_path = navigate.get_actual_path()
		self.path = path
		self.category = category
		self.trackswav = trackswav
		self.legend = legend
		self.colormesh = colormesh
		self.corresponding_cat_array = corresponding_cat_array
		self.dest = dest
		self.dpi = dpi
		self.width = width
		self.height = height
		print("------------------------------------------------------------------")
		print("The ",name," frame's datas are : ")
		print("The path of the wav directory containing the songs : ",path)
		print("The category selected is : ",category)
		print("This directory contains ",len(trackswav)," songs.")
		print("State of the legends :  ",legend)
		print("Is the graph will be colormesh :  ",colormesh)
		print("------------------------------------------------------------------")

	def run(self) :
		print("Running the Thread : ",self.name)
		cmpt = 0
		for i, item in enumerate(self.trackswav) :
			if(cmpt>=50) :
				break
			if(item[-3:]=="wav") :
				if not os.path.exists(self.my_json["dataset_image_path"]+"\\"+self.category+"\\"+item.replace(".wav", ".png")) :
					try :
						sample_rate, samples = wavfile.read(str(self.my_json['dataset_wav_path']+'\\'+self.category+'\\'+item))
						# samples, sample_rate = lr.load(str(self.my_json['dataset_wav_path']+'\\'+self.category+'\\'+item), mono=True)
						duration_seconds = len(samples)/sample_rate # 44100 Hz -> 30s.
						nfft = 2048 # CARE ABOUT THE RESOLUTION -> loss of precision. basis = None
						noverlap = 128 # number of frequency level. default = 128
						spect_Fs = sample_rate/2 
						try :
							fig = plt.figure(frameon=False, figsize=(2,2)) #, figsize=(self.height/self.dpi, self.width/self.dpi), dpi=self.dpi)
							ax = plt.axes()
							plt.ioff()
							if(self.colormesh) :
								f, t, Sxx = signal.spectrogram(samples, sample_rate, return_onesided=False)
								plt.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(Sxx, axes=0))
							else : 
								plt.specgram(samples, NFFT=nfft, noverlap=noverlap, Fs=spect_Fs, Fc=0, cmap='gray_r', sides='default', mode='default', scale='dB')
							try :
								if(self.legend) :
									plt.colorbar(None, use_gridspec=True)
									plt.title('Spectrogram of '+item)
									plt.ylabel('Frequency [Hz]')
									plt.xlabel('Time [sec]')
								else : 
									frame = plt.gca()
									frame.axes.get_xaxis().set_visible(False)
									frame.axes.get_yaxis().set_visible(False)
									ax.set_axis_off()
								fig.savefig(str(self.my_json["dataset_image_path"]+"\\"+self.category+"\\"+item.replace("wav","png")), frameon="false", bbox_inches='tight', transparent=True, pad_inches=0.0)
								print(self.name," Index ",i,". The spectrogram : ",item.replace('wav','png'), " was created.")
								try :
									im = Image.open(str(self.my_json["dataset_image_path"]+"\\"+self.category+"\\"+item.replace("wav","png")))
									try : 
										meta = PngImagePlugin.PngInfo()
										my_meta = {"length_secs" : str(duration_seconds), "category_shaped" : str(self.corresponding_cat_array), "category" : self.category, "overlap" : str(noverlap), "nfft" : str(nfft), "Fs" : str(spect_Fs), 'Frequency' : str(sample_rate)}
										for x in my_meta: 
										    meta.add_text(x, my_meta[x])
										im.save(str(self.my_json["dataset_image_path"]+"\\"+self.category+"\\"+item.replace("wav","png")), "png", pnginfo=meta)
									except Exception as inst :
										print(inst)
										print(self.name," Index ",i,". Error creating the metadatas.")	
								except Exception as inst :
									print("\nException : ",inst)
									print(self.name," Index ",i,". The png image representing the spectrogram can't be opened.")
							except : 
								print(self.name," Index ",i,". The spectrogram : ",item.replace('wav','png'), " can't be created. Error creating/hiding the legend of the spectrogram.")
							plt.close()
						except Exception as inst :
							print("Exception : ",inst) 
							print(self.name," Index ",i,". The spectrogram : ",item.replace('wav','png'), " can't be created. Error creating the spectogram.") 
						cmpt +=1 
					except :
						print(self.name," Index ",i,". The spectrogram : ",item.replace('wav','png'), " can't be created. Error reading the data from the wav file and extracting the frames.") 
				else :
					print(self.name," Index ",i,". The spectrogram : ",item.replace('wav','png'), " already exists.") 
			else : 
				print(self.name," : Error exporting creating the png. the ",item," isn't at the good format.\n")
		print("The thread : ",self.name, " is terminated.")

class ImageProcessGTZAN() :
	"""
		overview : Thread which permite filling the database.
		We have 2k items, and it takes an average of 5 seconds per item.
		We will use Threads to make it faster.
	"""
	def __init__(self, name, my_json, path, category, trackswav, dest, dpi, height, width, legend=False) :
		print("Creating the Multiprocess class ",name)
		self.name = name
		self.my_json = my_json
		self.old_path = navigate.get_actual_path()
		self.path = path
		self.category = category
		self.trackswav = trackswav
		self.dest = dest
		self.dpi = dpi
		self.height = height
		self.width = width
		self.legend = legend
		print("------------------------------------------------------------------")
		print("The ",name," frame's datas are : ")
		print("The path of the wav directory containing the songs : ",path)
		print("The category selected is : ",category)
		print("This directory contains ",len(trackswav)," songs.")
		print("State of the legends :  ",legend)
		print("------------------------------------------------------------------")

	def run(self) :
		print("Running the Thread : ",self.name)
		for i, item in enumerate(self.trackswav) :
			if(item[-3:]=="wav") :
				if not os.path.exists(self.my_json["dataset_images_gtzan_path"]+"\\"+self.category+"\\"+item.replace(".wav", ".png")) :
					try :
						samples, sample_rate = lr.load(str(self.my_json['dataset_wav_gtzan_path']+'\\'+self.category+'\\'+item), mono=True, duration=5)
						duration_seconds = len(samples)/sample_rate # 44100 Hz -> 30s.
						nfft = 2048 # CARE ABOUT THE RESOLUTION -> loss of precision. basis = None
						noverlap = 128 # number of frequency level. default = 128
						Fs =  2
						try :
							fig = plt.figure(figsize=(self.height/self.dpi, self.width/self.dpi), dpi=self.dpi)
							ax = plt.axes()
							plt.ioff()
							# plt.specgram(samples, NFFT=nfft, noverlap=noverlap, Fs=Fs, Fc=0, cmap='gray_r', sides='default', mode='default', scale='dB')
							mel = lr.feature.melspectrogram(y=samples, sr=sample_rate, n_fft=2048, n_mels=128, hop_length=512)
							lr.display.specshow(librosa.power_to_db(mel, ref=np.max))
							try :
								if(self.legend) :
									plt.colorbar()
									plt.title('Spectrogram of '+item+' in dB.')
									plt.ylabel('Frequency [Hz]')
									plt.xlabel('Time [sec]')
								else : 
									ax.set_axis_off()
								fig.savefig(str(self.my_json["dataset_images_gtzan_path"]+"\\"+self.category+"\\"+item.replace("wav","png")), bbox_inches='tight', transparent=True, pad_inches=0.0)
								print(self.name," Index ",i,". The spectrogram : ",item.replace('wav','png'), " was created.")
							except : 
								print(self.name," Index ",i,". The spectrogram : ",item.replace('wav','png'), " can't be created. Error creating/hiding the legend of the spectrogram.")
							plt.close()
						except Exception as inst :
							print("Exception : ",inst) 
							print(self.name," Index ",i,". The spectrogram : ",item.replace('wav','png'), " can't be created. Error creating the spectogram.") 
					except Exception as inst :
						print(self.name," Index ",i,". The spectrogram : ",item.replace('wav','png'), " can't be created. Error reading the data from the wav file and extracting the frames.") 
						print("Exception : ",inst)
				else :
					print(self.name," Index ",i,". The spectrogram : ",item.replace('wav','png'), " already exists.") 
			else : 
				print(self.name," : Error exporting creating the png. the ",item," isn't at the good format.\n")
		print("The thread : ",self.name, " is terminated.")

class SortMP3Thread(Thread) :
	def __init__(self, name, my_json, path, category, tracksmp3, datas_csv) :
		Thread.__init__(self)
		print("Creating the Thread ",name)
		self.name = name
		self.my_json = my_json
		self.old_path = navigate.get_actual_path()
		self.path = path
		self.category = category
		self.tracksmp3 = tracksmp3
		self.datas_csv = datas_csv

	def run(self) :
		print("Running the Thread : ",self.name)
		for i, mp3 in enumerate(self.tracksmp3) :
			if(mp3[-3:]=="mp3") :
				flag = False
				track_id = int(mp3[:-4])
				mp3_metadatas = songs.getSongInfo(self.my_json["dataset_mp3_path"]+"\\"+self.category+"\\"+mp3)
				if("TIT2" in mp3_metadatas and "TRCK" in mp3_metadatas) :
						flag = True
						mp3_piste = mp3_metadatas['TRCK']
						mp3_title = mp3_metadatas["TIT2"]
						genre = self.datas_csv.loc[self.datas_csv["title"] == mp3_title]
						if(genre["genre_top"].size>=2) :
							genre = genre.loc[genre["number"] == mp3_piste]
						if(genre.empty or genre["genre_top"].size>=2 or isinstance(genre["genre_top"].item(), float)) :
							print(self.name," : Impossible to copy : ",mp3,". The corresponding DataFrame is empty.")
						else :
							if not(os.path.exists(self.my_json['dataset_mp3_sorted_path']+'\\'+genre["genre_top"].item().replace("/", "_"))) :
								os.makedirs(self.my_json['dataset_mp3_sorted_path']+'\\'+genre["genre_top"].item().replace("/", "_"))
								print(self.name," : ",genre["genre_top"].item().replace("/", "_")," folder created.")
							
							if not os.path.exists(self.my_json["dataset_mp3_sorted_path"]+'\\'+genre["genre_top"].item().replace("/", "_")+"\\"+mp3) :
								shutil.copy(self.my_json["dataset_mp3_path"]+"\\"+self.category+"\\"+mp3, self.my_json["dataset_mp3_sorted_path"]+"\\"+genre["genre_top"].item().replace("/", "_")+"\\"+mp3)
								print(self.name," : ",mp3," song was copied into MP3Sorted .")
							else : 
								print(self.name," : ",mp3," already exists. Can't copy it.")
				else : 
					print(self.name," : Error -> There isn't all the datas we need in ",mp3," metadatas.")
			else : 
				print(self.name," : Error sorting the ",mp3," file/directory. It's not a mp3 song.")
		print("The thread : ",self.name, " is terminated.")