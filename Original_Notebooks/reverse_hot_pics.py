###read in one-hot images of proteins and turn them back into strings of AA letter###

import os
from PIL import Image
import numpy as np
from glob import glob
from scipy.misc import imread, imresize, bytescale
import csv

folder = '/home/whale/Desktop/Rachel/DeepProteins/AutoAntibodies/cyclegan/results/anti_cyclegan/test_latest/images/'
os.chdir(folder)

def read(kind):
	imsz = (26, 300)
	files = '*' + kind + '.png'
	names = glob(files) 
	for idx, name in enumerate(names):
		img = imread(name)
		img = img[:, :, 0]
		img = imresize(img, imsz)
		vec = np.argmax(img, axis=0)
	
		news = []
		for bit in vec:	
			bit = chr((bit + 97))
			if bit == 'x':
				continue
			news.append(bit)
	
		news = list(map(lambda x:x.upper(),news))
		news = ''.join(news)
		print(news)

		saver = folder+ kind + '/' + name
		print(saver)
		path = saver + '.csv'
		with open(path, mode='w', newline='') as saver:
			writer = csv.writer(saver, delimiter=',')
			writer.writerow(news)
		print('meowzers')

read('rec_B')
read('rec_A')




