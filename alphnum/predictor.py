#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import ssl
from pprint import pprint

import cv2 as cv
import numpy as np
import tensorflow as tf
from keras import applications
from keras.models import model_from_json
from keras.preprocessing import image


class ImageHelper:
    @staticmethod
    def get_image_by_path(image_path, target_size=None):
        """
        Given a path to an image, this function loads the stored image and returns it as PIL-Format
        (https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/load_img)
        :param target_size:
        :return: PIL-Format
        """
        img = image.load_img(image_path, target_size=target_size)
        return img

    @staticmethod
    def binarize(image_path):
        """
        Binarize image.

        :param image_path:
        :return:
        """
        img = cv.imread(image_path, 0)
        ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
        image_path_binary = os.path.splitext(image_path)[0] + "_binary" + os.path.splitext(image_path)[1]
        cv.imwrite(image_path_binary, thresh1)
        return image_path_binary

    @staticmethod
    def blurr(image_path):
        img = cv.imread(image_path, 0)
        blur = cv.GaussianBlur(img, (5, 5), 0)
        image_path_blurred = os.path.splitext(image_path)[0] + "_blur" + os.path.splitext(image_path)[1]
        cv.imwrite(image_path_blurred, blur)
        return image_path_blurred


class ModelHelper:
    @staticmethod
    def load_model(filename_weight, filename_model):
        """
        Given a filename for the weight and the model structure this function returns the correspondent model.
        :param filename:
        :return: keras model
        """
        with open(filename_model, 'r') as file:
            model = model_from_json(file.read())
            file.close()

        model.load_weights(filename_weight)
        return model

    @staticmethod
    def load_class_index(filename_class_index):
        """
        Given the filename of a class_index, this function loads the stored file and returns it as dict.
        :param filename_class_index:
        :return: dict
        """
        class_dictionary = np.load(filename_class_index).item()
        return class_dictionary


class Predictor:
    def __init__(self):
        self.graph = None
        self.top_model_path = 'model/top_model'
        self.class_index_path = 'model/class_indices.npy'
        self.target_size = (224, 224)
        self.model_base = self.get_model_bottleneck()
        self.model_top = self.get_top_model()
        self.class_index = ModelHelper.load_class_index(self.class_index_path)
        self.graph = tf.get_default_graph()

    def get_top_model(self):
        """
        Returns the fc- model block that is used to classify the features.
        :return: keras model
        """
        model = ModelHelper.load_model(filename_weight=self.top_model_path + '.h5',
                                       filename_model=self.top_model_path + '.json')

        return model

    def get_model_bottleneck(self):
        """
        Return model ready to extract features from images.
        :return: keras.model
        """
        ssl._create_default_https_context = ssl._create_unverified_context
        model = applications.VGG16(include_top=False, weights='imagenet')
        return model

    def preprocess_input(self, x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

    def decode_prediction(self, prediction):
        """
        Given a prediction array, the correspondent class is returned.
        :param index:
        :return:
        """
        index = np.argmax(prediction)

        inv_map = {v: k for k, v in self.class_index.items()}
        label = inv_map[index]
        return label, np.amax(prediction)

    def predict_image(self, image_paths, binarize=False, blur=False):
        """
        Given a path to an image, this function returns the correspondent imagel-label and the probability.
        :param image_path:
        :return: dict
        """
        predictions = list()
        for image_path in image_paths:
            if blur:
                image_path = ImageHelper.blurr(image_path)
            if binarize:
                image_path = ImageHelper.binarize(image_path)

            img = ImageHelper.get_image_by_path(image_path, self.target_size)

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = self.preprocess_input(x)

            with self.graph.as_default():
                features = self.model_base.predict(x)
                preds = self.model_top.predict(features)
            label, probability = self.decode_prediction(preds)

            predictions.append({"image_path": image_path,
                                "label": label,
                                "probability": probability})
        return predictions


if __name__ == '__main__':
    image_paths = ["dataset/test/seven/52077.jpg",
                   "dataset/6_bold.jpg",
                   "dataset/9_draw.jpg",
                   "dataset/4_draw.jpg"]

    predictor = Predictor()
    prediction = predictor.predict_image(image_paths=image_paths, binarize=True, )
    pprint(prediction)
