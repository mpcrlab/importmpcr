import os
from glob import glob
import imageio
import numpy as np
import matplotlib.pyplot as plt

img_path = '/home/vmlubuntu/bird/paddedimgs'
save_path = '/home/vmlubuntu/bird/stackedimgs/'
num_stacked = 2
img_h = 256
img_w = 1200


#sorting_ *this is done.. official sorted list = master______________________________________________________________________
os.chdir(img_path)
names = sorted(glob('*.png'))
names.sort()
num_names = len(names)

bin = []
number = []
folders = []
for file in names:
  split_name = file.split('-')
  folder = split_name[0]
  folders.append(folder)
  filename = split_name[1]
  bin.append(filename)
  filename = filename.split('.')
  filename = filename[0]
  number.append(filename)
number.sort(key=float)
songs = []
for song in number:
    file = str(song) + '.png'
    songs.append(file)
master = []
for song in songs:
    idx = (bin.index(song))
    name = str(folders[idx]) + '-' + song
    master.append(name)


#stacking___(three and guess fourth)___________________________________________________________________
init = 0
count = 2

#Class A = labels/targets
read in first 4 songs from master and stack
- do this in a sliding fashion
- save in a folder



#Class B = predictions
read in first 3 songs from master and stack, with empty position for 4
- do this in a sliding fashion
-save stacks somewhere

init += 1

stacked = np.zeros([num_stacked, img_h, img_w, 3])
for idx, song in enumerate(songs):
    img = imageio.imread(song)
    stacked[idx, ...] = img.astype(np.uint8)
    if idx == num_stacked:
        break

img = np.concatenate((stacked[0,:,:,:], stacked[1,:,:,:]))
img = np.concatenate((img, stacked[2,:,:,:]))

name = savepath + song
imageio.imwrite(name, img.astype(np.uint8))
