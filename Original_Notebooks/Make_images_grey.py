import os
from PIL import Image
import numpy as np
from skimage import io
import imageio
from skimage.color import rgb2gray

os.chdir('/home/mpcrlab/instagan/datasets/anti2auto')
counts = {}

def check_channels(folder):
  files = os.listdir(folder)
  for filename in files:
    filepath = os.path.join(folder, filename)
    im = np.asarray(Image.open(filepath)) # read in the image
    if im.shape == (256,256,1):
      print('passed')
    else:
      im = rgb2gray(im)
      imageio.imwrite(filepath, im)
      print('fixed')
  print('done.')


# check_channels('trainA')
# check_channels('trainB')
check_channels('trainA_seg')
check_channels('trainB_seg')
