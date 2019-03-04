name='google_images_download'

import subprocess

def install(name):
    subprocess.call(['pip', 'install', name])

install(name)


from google_images_download import google_images_download
import shutil
import os
import cv2



