import argparse
import os
import time

# Hide all tensorflow I, W, E messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Save plot in background
matplotlib.use('Agg')

from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Images dimensions
img_height, img_width = 150, 150

# Config model
num_train_samples, num_validation_samples = 3000, 2000
epochs = 30
batch_size = 128
class_mode = 'categorical'

# Initialize the model
model = Sequential()


def load_data(data_dir='data'):
    # Look for sub directories in data
    train_data_dir = os.path.join(data_dir, 'train')
    test_data_dir = os.path.join(data_dir, 'test')

    # Construct the image generator for data augmentation
    train_datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1. / 255,
        shear_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=class_mode
    )

    validation_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=class_mode
    )

    print('DATA LOADED!')

    return train_generator, validation_generator


def build_model(class_count):
    # Add first (intput) layer
    model.add(Conv2D(32, (3, 3), input_shape=(img_height, img_width, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Add hidden layers
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))

    # Add final (output) layer
    model.add(Dense(class_count, activation='softmax'))

    model.compile(
        loss=class_mode + '_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    print('MODEL SUCCESSFULLY COMPILED!')


def train_model(train_data, test_data):
    start = time.process_time()

    # Train the network
    history = model.fit_generator(
        train_data,
        steps_per_epoch=num_train_samples // batch_size,
        epochs=epochs,
        validation_data=test_data,
        validation_steps=num_validation_samples // batch_size,
    )

    end = time.process_time()
    train_time = (end - start) / 60

    print(f'TRAINING COMPLETED IN: {train_time:.2f} minutes')

    return history


def plot_training_progress(history, plot_dir):
    plt.style.use('ggplot')
    plt.figure()

    N = np.arange(epochs)
    plt.plot(N, history.history['loss'], label='train_loss')
    plt.plot(N, history.history['val_loss'], label='val_loss')
    plt.plot(N, history.history['accuracy'], label='train_acc')
    plt.plot(N, history.history['val_accuracy'], label='val_acc')

    plt.title('Training loss and accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.legend(loc='lower left')
    plt.savefig(plot_dir)


def save_model(save_path='model.h5'):
    # Check file extension
    if save_path[-3:] != '.h5':
        save_path += '.h5'

    model.save(save_path)

    print('MODEL SUCCESSFULLY SAVED!')


def load_trained_model(model_path):
    trained_model = load_model(model_path)

    print('MODEL LOADED!')

    return trained_model


def evaluate(test_data):
    score = model.evaluate(test_data, verbose=0)
    evaluation = dict()
    for i in range(len(model.metrics_names)):
        evaluation[model.metrics_names[i]] = score[i]

    print('EVALUATION METRICS:')
    for key in evaluation:
        print(f'+ {key}: {evaluation[key] * 100}%')


def predict(img_path, classes):
    # Open the image and store it in a numpy array
    img = load_img(img_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = img.astype('float32') / 255
    img = np.expand_dims(img, axis=0)

    # Classify input image
    prediction = model.predict(img)[0]
    classes = {value: key for key, value in classes.items()}
    result = dict()
    for i in range(len(classes)):
        result[classes[i]] = list(prediction)[i]

    print('PREDICTING:')
    for label in result:
        print(f'+ It is {label} with probability of {result[label] * 100}%')


if __name__ == '__main__':
    # Construct the argument parse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', metavar='DATA', type=str, required=True,
                        help='directory to the input dataset')
    parser.add_argument('-s', '--save', metavar='SAVE', type=str,
                        help='save the trained model')
    parser.add_argument('-l', '--load', metavar='LOAD', type=str,
                        help='load an existing model')
    parser.add_argument('-i', '--img', metavar='IMAGE', type=str, required=True,
                        help='name of the image for predicting')
    args = vars(parser.parse_args())

    train_data, test_data = load_data(args['dataset'])
    classes = train_data.class_indices

    if not args['load']:
        build_model(len(classes))
        history = train_model(train_data, test_data)
        plot_training_progress(history, 'training-plot.png')

        if not args['save']:
            while True:
                ans = input('Do you want to save your model? [y/n] ').lower()
                if ans in ['yes', 'y']:
                    save_model()
                    break
                if ans in ['no', 'n']:
                    break
        else:
            save_model(args['save'])

    else:
        model = load_trained_model(args['load'])

    evaluate(test_data)
    predict(args['img'], classes)
