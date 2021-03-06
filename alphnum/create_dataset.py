#!/usr/bin/python3

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


def extract_and_split_jpgs(train=0.7, validation=0.2, test=0.1):
	"""
	Extract the images and split them into percentage accordingly to the parameters into
	train, validation and test-sets.

	:param train:
	:param validation:
	:param test:
	:return:
	"""
	for label in label_dirs.values():
		for type in ["training", "validation", "test"]:
			path = os.path.join("dataset", type, label)
			if not os.path.exists(path):
				os.makedirs(path)

	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	label_idx_dict = {label: [] for label in label_dirs.keys()}  # init dict with empty lists.

	for idx, label in enumerate(y_train):
		label_idx_dict[str(label)].append(idx)

	threshold_training = train * 6000
	threshhold_validation = (validation + train) * 6000

	# Now label_idx_dict looks like this : {"5": [0, 4, 12, 510, 601, 23], "2" = [...], ...}
	for label, idx_list in tqdm(label_idx_dict.items()):
		for idx, element in enumerate(idx_list):
			if idx < threshold_training:
				path = os.path.join("dataset", "training", label_dirs[label])
			elif (threshold_training <= idx) and (idx < threshhold_validation):
				path = os.path.join("dataset", "validation", label_dirs[label])
			else:
				path = os.path.join("dataset", "test", label_dirs[label])

			nparray_inv = np.invert(np.array(x_train[element], dtype='uint8'))
			filename = '{:05d}'.format(element) + ".jpg"
			cv2.imwrite(os.path.join(path, filename), nparray_inv)


if __name__ == "__main__":
	extract_and_split_jpgs()
