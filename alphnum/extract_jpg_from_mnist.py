#!/usr/bin/python3

"""
This script extracts the images (jpg.) from the mnist dataset provided by keras.
"""

import os

import cv2
import numpy as np
from keras.datasets import mnist
from tqdm import tqdm

label_dirs = {
	"0": "zero",
	"1": "one",
	"2": "two",
	"3": "three",
	"4": "four",
	"5": "five",
	"6": "six",
	"7": "seven",
	"8": "eight",
	"9": "nine"
}

for label in label_dirs.values():
	path = "dataset/" + label
	if not os.path.exists(path):
		os.makedirs(path)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

for i in tqdm(range(60000)):
	label = label_dirs[str(y_train[i])]
	path = os.path.join("dataset", label)
	nparray_inv = np.invert(np.array(x_train[i], dtype='uint8'))
	filename = '{:05d}'.format(i) + ".jpg"
	cv2.imwrite(os.path.join(path, filename), nparray_inv)
