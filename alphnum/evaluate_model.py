#!/usr/bin/python3

import os
from os import listdir
from os.path import isfile, join

from predictor import Predictor

labels = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero"]


def get_all_files_from(path):
    files = [os.path.join(path, f) for f in listdir(path) if isfile(join(path, f))]
    return files

def evaluate(folder):
    predictor = Predictor()

    for label in labels:
        path = os.path.join(folder, label)
        filenames = get_all_files_from(path)


        predictions = predictor.predict_image(filenames)
        i = 0
        for pred in predictions:
            if pred["label"] == label:
                i += 1
        print("Label", label, ":", str(i/float(len(filenames))))


if __name__ == '__main__':
    evaluate("handwritten_digits")
    evaluate("handwritten_digits_binary")
    evaluate("handwritten_digits_resized_binary")
