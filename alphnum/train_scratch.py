import os
import pickle

import matplotlib.pyplot as plt
from keras import backend as K
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

# dimensions of our images.
img_width, img_height = 28, 28

# set paths
train_data_dir = 'dataset/training'
validation_data_dir = 'dataset/validation'
top_model_path = 'model/scratch_model'
history_path = 'training_100epochs/training/training_mnist_scratch.json'
class_indices_path = 'model/class_indices_scratch.npy'

# train parameters.
epochs = 100
batch_size = 10
learning_rate = 0.0001


# Create folders
def create_folders():
    folders = ["model", "bottleneck_features", "training", "dataset"]
    for folder in folders:
        if not os.path.exists(folder):
            os.mkdir(folder)


# Persist history of training stage.
def save_history(history, filename):
    print("Persist training history in", filename)
    with open(filename, 'wb') as outfile:
        pickle.dump(history.history, outfile)
        outfile.close()


# Persist model structure and its weights.
def save_model(model, filename):
    print("Persist model completely in", filename)
    with open(filename + '.json', 'w') as outfile:
        outfile.write(model.to_json(sort_keys=True,
                                    indent=4,
                                    separators=(',', ': ')))
        outfile.close()

    # Save weights
    model.save(filename + '.h5')


# Extract the bottleneck features of the training and validation samples.
def train_scratch():
    datagen = ImageDataGenerator(rescale=1. / 255)
    datagen2 = ImageDataGenerator(rescale=1. / 255)

    train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    validation_generator = datagen2.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    # Number of training samples and Number of classes.
    nb_train_samples = len(train_generator.filenames)
    nb_validation_samples = len(validation_generator.filenames)
    num_classes = len(train_generator.class_indices)
    model = create_top_model(num_classes=num_classes)

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)


def plot_loss(history):
    plt.figure(1)

    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.subplot(212)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def create_top_model(num_classes):
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    opt = optimizers.RMSprop(lr=learning_rate)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    create_folders()
    train_scratch()
# history = pickle.load(open(history_path, "rb"))
# plot_loss(history)
