#!/usr/bin/python3

import os
from util import Util
from predictor import Predictor

labels = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero"]



def evaluate(folder):
    predictor = Predictor()

    for label in labels:
        path = os.path.join(folder, label)
        filenames = Util.get_all_files_from(path)


        predictions = predictor.predict_image(filenames)
        i = 0
        for pred in predictions:
            if pred["label"] == label:
                i += 1
        print("Label", label, ":", str(i/float(len(filenames))))


if __name__ == '__main__':
    evaluate("handwritten_kulli")
    evaluate("handwritten_kulli_binary")
    evaluate("handwritten_kulli_resized_binary")
