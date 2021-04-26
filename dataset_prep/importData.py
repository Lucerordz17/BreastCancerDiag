import os
import cv2
#import PIL # optional
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

GooglePath = "/content/gdrive/MyDrive/Breast Cancer Data"
savepath = 'data'
images_path1 = "/content/gdrive/MyDrive/Breast Cancer Data/train"
images_path2 = "/content/gdrive/MyDrive/Breast Cancer Data/test"

path1 = os.listdir(images_path1)
for image in enumerate(path1):
    image_cv2 = cv2.imread(os.path.join(images_path1, image))
    cv2.imwrite(os.path.join(savepath, 'train', image), image_cv2)

path2 = os.listdir(images_path2)
for image in enumerate(path2):
    image_cv2 = cv2.imread(os.path.join(images_path2, image))
    cv2.imwrite(os.path.join(savepath, 'test', image), image_cv2)

f1 = open(GooglePath+"test_split_v3.txt", "r")
contents1 = f1.open()
test_file = open("test_split_v3.txt",'w')
test_file.write(contents1)
test_file.close()

f2 = open(GooglePath+"train_split_v3.txt", "r")
contents2 = f2.open()
train_file = open("train_split_v3.txt", 'w')
train_file.write(contents2)
train_file.close()