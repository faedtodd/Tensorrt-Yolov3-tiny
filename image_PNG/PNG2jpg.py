from __future__ import print_function
import numpy as np
import cv2
from PIL import ImageDraw
from PIL import Image
import os
import sys

path='./'

for filename in os.listdir(path):
    if os.path.splitext(filename)[1]=='.PNG':
        img = cv2.imread(path + filename)
        new_filename = filename.replace('.PNG', '.jpg')
        cv2.imwrite(path + new_filename, img)


#image = cv2.imread('./2.PNG')
#cv2.imwrite('./2.jpg', image)
