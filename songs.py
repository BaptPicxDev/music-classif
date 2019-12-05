#############################################
# author : Baptiste PICARD 		 			#
# date : 21/08/2019				 			#
# for : UNcuyo PFE 							#
# last change : 26/08/2019					#
# review : Creating 2 functions				#
# 		- wavToSpectrogram					#
# 		- saveSpectrogram					#
# 								 			#
# overview : This script helps to play and  #
# transform songs into images.				#
# The next step is to handle the 			#
# images (related to songs) and learn from 	#
# those.									#
#############################################

# All the imports
# My scripts
# Librairies
import os 
import utils
import navigate
from librosa import * 
from scipy import *
import numpy as np
from scipy import signal
from scipy.fftpack import fft
from scipy.io import wavfile # Read .wav files
import matplotlib.pyplot as plt # Draw figures
from pygame import mixer # To play .mp3 songs
from mutagen.id3 import ID3 # To get the metadatas
from PIL import Image # Slice images each 30s.

# Functions.
def playsongmp3(path_of_song) : 
	"""
		overview : play a .mp3 song
		inputs :
			- path_of_song : string which represents the path of the song i want to play.
	"""
	mixer.init()
	mixer.music.load(path_of_song) # r before to tranform the string in 'raw string'
	print("Song name ",name_of_song," is playing.")
	mixer.music.play()
	while(mixer.music.get_busy()) :
		pass

def getSongInfo(path_of_song) : 
	"""
		overview : play a .mp3 song
		inputs :
			- path : string which represents the path of the song i want to play.
			- song_name : string chich represents the name of the song.
		output : 
			- song_metadatas : dictionary of informations about the song.
	"""
	song_metadata = ID3(path_of_song)
	return song_metadata

def getSignalTimeDomain(my_json, path_of_song, legend=True) : # -> 3s for each.
	print("Getting the signal in time domain.")
	song_name = path_of_song.split("\\")[-1].replace(".wav", "")
	sample_rate, samples = wavfile.read(path_of_song)
	duration_seconds = len(samples)/sample_rate
	time_array = np.arange(0, duration_seconds, 1/sample_rate)
	plt.figure()
	ax = plt.axes()
	plt.plot(time_array, samples)
	if(legend==True) :
		plt.ylabel("Amplitude")
		plt.xlabel("Time (Seconds)")
		plt.title("Time domain - Fe = {} Hz, Ts = {} seconds".format(sample_rate, int(duration_seconds)))
	else : 
		plt.ioff()
		frame = plt.gca()
		frame.axes.get_xaxis().set_visible(False)
		frame.axes.get_yaxis().set_visible(False)
		ax.set_axis_off()
	plt.savefig(my_json['results_folder']+'\\'+song_name+'_test_time.png')				
	plt.close()

def getSignalFrequencyDomainAll(my_json, path_of_song, legend=True) : # -> 103 s
	print("Getting the signal in frequency domain - All the frequency domains.")
	song_name = path_of_song.split("\\")[-1].replace(".wav", "")
	sample_rate, samples = wavfile.read(path_of_song)
	duration_seconds = len(samples)/sample_rate
	freq_array = np.arange(0, samples.size, sample_rate)
	FFT = abs(fft(samples))
	plt.figure() 
	ax = plt.axes()
	plt.plot(FFT)
	if(legend==True) :
		plt.ylabel("Amplitude")
		plt.xlabel("Frequence (Hertz)")
		plt.title("Frequency domain - Fe = {} Hz, Ts = {} seconds".format(sample_rate, int(duration_seconds)))
	else : 
		plt.ioff()
		frame = plt.gca()
		frame.axes.get_xaxis().set_visible(False)
		frame.axes.get_yaxis().set_visible(False)
		ax.set_axis_off()
	plt.savefig(my_json["results_folder"]+'\\'+song_name+'_test_all_freqs.png')				
	plt.close()

def getSignalFrequencyDomainOneSide(my_json, path_of_song, legend=True) :
	print("Getting the signal in frequency domain - One side frequency domains.")
	song_name = path_of_song.split("\\")[-1].replace(".wav", "")
	sample_rate, samples = wavfile.read(path_of_song)
	duration_seconds = len(samples)/sample_rate
	freq_array = np.arange(0, samples.size, sample_rate)
	FFT = abs(fft(samples))	
	FFT_side = FFT[range(int(samples.size/2))] # one side FFT range
	plt.figure() # Fig 2 -> One side frequecy domains
	ax = plt.axes()
	plt.plot(FFT_side)
	if(legend==True) :
		plt.ylabel("Amplitude")
		plt.xlabel("Frequence (Hertz)")
		plt.title("Frequency domain - One side.")
	else : 
		plt.ioff()
		frame = plt.gca()
		frame.axes.get_xaxis().set_visible(False)
		frame.axes.get_yaxis().set_visible(False)
		ax.set_axis_off()
	plt.savefig(my_json["results_folder"]+'\\'+song_name+'_test_one_side_freqs.png')			
	plt.close()

def getColormesh(my_json, path_of_song, legend=True) :
	print("Getting the colormesh.")
	song_name = path_of_song.split("\\")[-1].replace(".wav", "")
	sample_rate, samples = wavfile.read(path_of_song)
	f, t, Sxx = signal.spectrogram(samples, sample_rate, return_onesided=False)
	plt.figure()
	ax = plt.axes()
	plt.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(Sxx, axes=0))
	if(legend==True) :
		plt.colorbar(None, use_gridspec=True)
		plt.title("Colormesh spectrogram of the track.")
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time [sec]')
	else :
		plt.ioff()
		frame = plt.gca()
		frame.axes.get_xaxis().set_visible(False)
		frame.axes.get_yaxis().set_visible(False)
		ax.set_axis_off()
	plt.savefig(my_json["results_folder"]+'\\'+song_name+'_colormesh_test.png')
	plt.close()

def getSpectrogram(my_json, path_of_song, legend=True) :
	print("Getting the spectrogram.")
	song_name = path_of_song.split("\\")[-1].replace(".wav", "")
	sample_rate, samples = wavfile.read(path_of_song)
	duration_seconds = len(samples)/sample_rate # 44100 Hz -> 30s.
	nfft = None # CARE ABOUT THE RESOLUTION -> loss of precision. basis = 256
	noverlap = 128 # number of frequency level.
	spect_Fs = sample_rate/2
	plt.figure()
	ax = plt.axes()
	plt.specgram(samples, NFFT=nfft, noverlap=noverlap, Fs=spect_Fs, cmap='gray')
	if(legend==True) :
		plt.colorbar(None, use_gridspec=True)
		plt.title("Spectrogram of the track.")
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time [sec]')
	else :
		plt.ioff()
		frame = plt.gca()
		frame.axes.get_xaxis().set_visible(False)
		frame.axes.get_yaxis().set_visible(False)
		ax.set_axis_off()
	plt.savefig(my_json["results_folder"]+'\\'+song_name+'_spectrogram_test.png')
	plt.close()

def getSpectrogramLibrosa(my_json, path_of_song, legend=True) : 
	print("Getting the spectrogram using librosa.")
	song_name = path_of_song.split("\\")[-1].replace(".wav", "")
	y, sr = load(path_of_song)
	spect = librosa.feature.melspectrogram(y=y, sr=sr,n_fft=2048, hop_length=512)
	spect = librosa.power_to_db(spect, ref=np.max)
	plt.figure()
	ax = plt.axes()
	librosa.display.specshow(spect)
	if(legend==True) :
		plt.colorbar(None, use_gridspec=True)
		plt.title("Spectrogram of the track.")
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time [sec]')
	else :
		plt.ioff()
		frame = plt.gca()
		frame.axes.get_xaxis().set_visible(False)
		frame.axes.get_yaxis().set_visible(False)
		ax.set_axis_off()
	plt.savefig(my_json["results_folder"]+'\\'+song_name+'_spectrogram_librosa_test.png')
	plt.close()

def getSongsFigures(my_json, path_of_song, my_legend) : 
	path_of_song = str(path_of_song)
	getSignalTimeDomain(my_json, path_of_song, legend=my_legend)
	getSignalFrequencyDomainAll(my_json, path_of_song, legend=my_legend)
	getSignalFrequencyDomainOneSide(my_json, path_of_song, legend=my_legend)
	getColormesh(my_json, path_of_song, legend=my_legend)
	getSpectrogram(my_json, path_of_song, legend=my_legend)

def sliceImage(my_json, image_path) :
	image_name  = image_path.split("\\")[-1].replace(".png", "")
	im = Image.open(image_path) 
	n_cuts = int(int(im.info['length_secs'].split('.')[0])/30) # We cut a song in part of 30s.
	imgwidth, imgheight = im.size # 640 * 640
	start_width = 0
	start_height = 0
	end_height = imgheight 
	step = int(imgwidth/n_cuts)
	s=1
	for w in range(0, imgwidth, step) :
		box = (start_width, start_height, w+step, end_height)
		start_width += step
		slice_x = im.crop(box)
		if(os.path.exists(str(image_name+"_slice_"+str(s)+'.png'))) :
			print("The slice already exists.")
		else :
			slice_x.save(str(image_name+"_slice_"+str(s)+'.png'))
			print("Slice ",s," created. This slice correspond at ",box," (pixels).")
		s+=1

def getFreqOverlapNFFT(my_json) :
	navigate.set_path_to_dataset_image(my_json)
	datasets = utils.get_list_subs()
	navigate.set_path(datasets[0])
	figs = utils.get_list_subs()
	im = Image.open(navigate.get_actual_path()+'\\'+figs[0])
	return im.info["Fs"], im.info["nfft"], im.info["overlap"], im.info["Frequency"]