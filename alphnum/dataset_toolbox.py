import os
from os import listdir
from os.path import isfile, join

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def plot_binary_example():
    img = cv.imread('samples/9_draw.jpg', 0)
    titles = ['Original Image']
    images = [img]
    for i in range(8):
        threshhold = i * 25
        ret, thresh1 = cv.threshold(img, threshhold, 255, cv.THRESH_BINARY)
        images.append(thresh1)
        titles.append("Threshold: " + str(threshhold))

    for i in range(8):
        plt.subplot(2, 4, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


def get_all_files_from(path, types=[".png", ".jpg"]):
    files = [os.path.join(path, f) for f in listdir(path) if
             (isfile(join(path, f)) and (os.path.splitext(f)[1] in types))]
    return files


def binarize_threshold_folder(folder, threshold=75):
    labels = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero"]
    for label in labels:
        path = os.path.join(folder, label)
        filenames = get_all_files_from(path)
        print(filenames)
        for filename in filenames:
            img = cv.imread(filename, 0)
            ret, thresh1 = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)
            cv.imwrite(filename, thresh1)


def binarize_otsu_folder(folder):
    labels = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero"]
    for label in labels:
        path = os.path.join(folder, label)
        filenames = get_all_files_from(path)
        print(filenames)
        for filename in filenames:
            img = cv.imread(filename, 0)
            otsu(filename, filename)


def otsu(source, target):
    print(source)
    img = cv.imread(source, 0)
    blur = cv.GaussianBlur(img, (5, 5), 0)
    # find normalized_histogram, and its cumulative distribution function
    hist = cv.calcHist([blur], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.max()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(1, 256):
        p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
        q1, q2 = Q[i], Q[255] - Q[i]  # cum sum of classes
        b1, b2 = np.hsplit(bins, [i])  # weights
        # finding means and variances
        m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
        v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2
        # calculates the minimization function
        fn = v1 * q1 + v2 * q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    # find otsu's threshold value with OpenCV function
    ret, otsu = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imwrite(target, otsu)


def resize_folder(folder, width, height):
    labels = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero"]
    for label in labels:
        path = os.path.join(folder, label)
        filenames = get_all_files_from(path)
        for filename in filenames:
            img = cv.imread(filename, 0)
            resized_image = cv.resize(img, (width, height))
            cv.imwrite(filename, resized_image)


if __name__ == '__main__':
    binarize_otsu_folder(folder="handwritten_kulli_binary")
    binarize_otsu_folder(folder="handwritten_kulli_resized_binary")
    resize_folder(folder="handwritten_kulli_resized_binary", width=28, height=28)
