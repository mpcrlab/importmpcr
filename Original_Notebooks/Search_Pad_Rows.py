
import os
from PIL import Image
import numpy as np
from skimage import io
import imageio

os.chdir('/home/whale/Desktop/Rachel/DeepProteins/AutoAntibodies/cyclegan/data/')
counts = {}

def pad_rows(folder):
  files = os.listdir(folder)
  for filename in files:
  	filepath = os.path.join(folder, filename)
  	im = np.asarray(Image.open(filepath)) # read in the image
  	im = im[:, :256, :]
  	zeros = np.zeros([228, im.shape[1], 3], dtype=np.uint8)
  	im = np.concatenate((im, zeros), axis=0)
  	imageio.imwrite(filepath, im)
  	print('meowzers')


#pad_rows('testA')
#pad_rows('testB')
# pad_rows('trainA')
# pad_rows('trainB')
