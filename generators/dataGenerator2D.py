import numpy as np
import cv2
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras


def crop_top(img, percent=0.15):
    offset = int(img.shape[0] * percent)
    return img[offset:]


def central_crop(img):
    size = min(img.shape[0], img.shape[1])
    offset_h = int((img.shape[0] - size) / 2)
    offset_w = int((img.shape[1] - size) / 2)
    return img[offset_h:offset_h + size, offset_w:offset_w + size]


def process_image_file(filepath, top_percent, size, enhance=False):
    img = cv2.imread(filepath)
    img = crop_top(img, percent=top_percent)
    img = central_crop(img)
    img = cv2.resize(img, (size, size))
    if enhance:
        img = xray_enhance(img)
    return img


def xray_enhance(img):
    gaussian_3 = cv2.GaussianBlur(img, (3, 3), 10.0)
    unsharp_image = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0, img)
    return unsharp_image


_augmentation_transform = ImageDataGenerator(
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.9, 1.1),
    zoom_range=(0.85, 1.15),
    fill_mode='constant',
    cval=0.,
)

def apply_augmentation(img):
    img = _augmentation_transform.random_transform(img)
    return img

class BalanceMalDataset(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(
            self,
            dataset,
            is_training=True,
            batch_size=8,
            input_shape=(224, 224),
            n_classes=2,
            num_channels=3,
            mapping={
                'BENIGN': 0,
                'MALIGNANT': 1
            },
            shuffle=True,
            augmentation=apply_augmentation,
            mal_percent=0.3,
            top_percent=0.08,
            balancing=False,
            enhance=False,
            aug=False
    ):
        'Initialization'
        self.dataset = dataset
        self.is_training = is_training
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.num_channels = num_channels
        self.mapping = mapping
        self.shuffle = True
        self.mal_percent = mal_percent
        self.n = 0
        self.augmentation = augmentation
        self.top_percent = top_percent
        self.balancing = balancing
        self.mean = None
        self.std = None
        self.enhance = enhance
        self.aug = aug

        datasets = {'BENIGN': [], 'MALIGNANT': []}
        for l in self.dataset:
            datasets[l[2]].append(l)
        if self.balancing:
            self.datasets = [
                datasets['BENIGN'],
                datasets['MALIGNANT'],
            ]
        else:
            self.datasets = [
                datasets['BENIGN'] + datasets['MALIGNANT']
            ]
            # print(len(self.datasets[0]), len(self.datasets[1]))

        self.on_epoch_end()

    def __next__(self):
        # Get one batch of data
        batch_x, batch_y = self.__getitem__(self.n)
        # Batch index
        self.n += 1

        # If we have processed the entire dataset then
        if self.n >= self.__len__():
            self.on_epoch_end
            self.n = 0

        return batch_x, batch_y

    def __len__(self):
        return int(np.ceil(len(self.datasets[0]) / float(self.batch_size)))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            for v in self.datasets:
                np.random.shuffle(v)

    def include_statistics(mean, std):
        self.mean = mean
        self.std = std

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros(
            (self.batch_size, *self.input_shape,
             self.num_channels)), np.zeros(self.batch_size)

        batch_files = self.datasets[0][idx * self.batch_size:(idx + 1) *
                                                             self.batch_size]
        if self.balancing:
            # upsample malignant cases
            mal_size = max(int(len(batch_files) * self.mal_percent), 1)
            mal_inds = np.random.choice(np.arange(len(batch_files)),
                                          size=mal_size,
                                          replace=False)

            choices_indices = np.random.choice(np.arange(len(self.datasets[1])),
                                               size=mal_size,
                                               replace=False)
            mal_files = [self.datasets[1][i] for i in choices_indices]
            for i in range(mal_size):
                batch_files[mal_inds[i]] = mal_files[i]

        for i in range(len(batch_files)):
            sample = batch_files[i]

            x = process_image_file(sample[1],
                                   self.top_percent,
                                   self.input_shape[0], self.enhance)

            if self.is_training and hasattr(self, 'augmentation') and self.aug:
                x = self.augmentation(x)

            x = x.astype('float32') / 255.0
            if self.mean is not None and self.std is not None:
                x = (x - mean) / std

            y = self.mapping[sample[2]]

            batch_x[i] = x
            batch_y[i] = y

        return batch_x, keras.utils.to_categorical(batch_y, num_classes=self.n_classes)
