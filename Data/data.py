# INSTALL
#pip install pydicom
#pip install opencv-python
#pip install pillow # optional
#pip install pandas

import pydicom as dicom
import os
import cv2
#import PIL # optional
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


## Go to Dropbox
DropboxPath = '/Users/lucerorodriguez/Dropbox (ASU)/Breast Cancer Data/manifest-ZkhPvrLo5216730872708713142'
os.chdir(DropboxPath)
metadata = pd.read_csv('metadata.csv')
Calc_Test = pd.read_csv('calc_case_description_test_set.csv')
Mass_Test = pd.read_csv('mass_case_description_test_set.csv')
Calc_Train = pd.read_csv('calc_case_description_train_set.csv')
Mass_Train = pd.read_csv('mass_case_description_train_set.csv')


### Patients IDs
IDs = metadata['Subject ID']

# Specify the .dcm folder path
images_path = ["" for n, string in enumerate(metadata['File Location'])]

path = "/Users/lucerorodriguez/Dropbox (ASU)/Breast Cancer Data/manifest-ZkhPvrLo5216730872708713142"
path2 = metadata['File Location'].apply(str)
for n, string in enumerate(metadata['File Location']):
    images_path[n] = path + path2[n][1:]
images_path = np.asarray(images_path)
# make it True if you want in PNG format
PNG = True

# Specify the output jpg/png folder path
test_path = "/Users/lucerorodriguez/Dropbox (ASU)/Breast Cancer Data/test"
train_path = "/Users/lucerorodriguez/Dropbox (ASU)/Breast Cancer Data/train"

train = []
test = []

for n in range(6775):
    folder_path = os.listdir(images_path[n])
    if len(folder_path) >1:
        image = '1-1.dcm'
    else:
        image = folder_path[0]
        image = image.decode("utf-8")
    # for image1 in enumerate(folder_path):
    #     image = image1
    #     image = image[1]
    #     image = image.decode("utf-8")
    #     break
    ds = dicom.dcmread(os.path.join(images_path[n], image))
    pixel_array_numpy = ds.pixel_array
    image = IDs[n] + '.png'
    ## If is from test  go to test
    if IDs[n].find('Calc-Test') == 0:
        Find = Calc_Test['ROI mask file path'].str.find(IDs[n])
        if 0 in Find:
            index = Calc_Test.index[Calc_Test['ROI mask file path'].str.contains(IDs[n],case=False)].tolist()[0]
            if  Calc_Test['pathology'][index] == 'BENIGN_WITHOUT_CALLBACK':
                key = 'BENIGN'
            else:
                key = Calc_Test['pathology'][index]
            test.append([Calc_Test['patient_id'][index], IDs[n], key])
            cv2.imwrite(os.path.join(test_path, image), pixel_array_numpy)
    elif IDs[n].find('Mass-Test') == 0:
        Find = Mass_Test['ROI mask file path'].str.find(IDs[n])
        if 0 in Find:
            index = Mass_Test.index[Mass_Test['ROI mask file path'].str.contains(IDs[n],case=False)].tolist()[0]
            if  Mass_Test['pathology'][index] == 'BENIGN_WITHOUT_CALLBACK':
                key = 'BENIGN'
            else:
                key = Mass_Test['pathology'][index]
            test.append([Mass_Test['patient_id'][index], IDs[n], key])
            cv2.imwrite(os.path.join(test_path, image), pixel_array_numpy)
    ## then go to train
    elif IDs[n].find('Calc-Train') == 0:
        Find = Calc_Train['ROI mask file path'].str.find(IDs[n])
        if 0 in Find:
            index = Calc_Train.index[Calc_Train['ROI mask file path'].str.contains(IDs[n], case=False)].tolist()[0]
            if  Calc_Train['pathology'][index] == 'BENIGN_WITHOUT_CALLBACK':
                key = 'BENIGN'
            else:
                key = Calc_Train['pathology'][index]
            train.append([Calc_Train['patient_id'][index], IDs[n], key])
            cv2.imwrite(os.path.join(train_path, image), pixel_array_numpy)
    else:
        Find = Mass_Train['ROI mask file path'].str.find(IDs[n])
        if 0 in Find:
            index = Mass_Train.index[Mass_Train['ROI mask file path'].str.contains(IDs[n], case=False)].tolist()[0]
            if  Mass_Train['pathology'][index] == 'BENIGN_WITHOUT_CALLBACK':
                key = 'BENIGN'
            else:
                key = Mass_Train['pathology'][index]
            train.append([Mass_Train['patient_id'][index], IDs[n], key])
            cv2.imwrite(os.path.join(train_path, image), pixel_array_numpy)
    if n % 50 == 0:
        print('{} image converted'.format(n))

# export to train and test csv
# format as patientid, filename, label, separated by a space
train_file = open("train_split_v3.txt",'w')
for sample in train:
    if len(sample) == 4:
        info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + ' ' + sample[3] + '\n'
    else:
        info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + '\n'
    train_file.write(info)

train_file.close()

test_file = open("test_split_v3.txt", 'w')
for sample in test:
    if len(sample) == 4:
        info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + ' ' + sample[3] + '\n'
    else:
        info = str(sample[0]) + ' ' + sample[1] + ' ' + sample[2] + '\n'
    test_file.write(info)

test_file.close()
