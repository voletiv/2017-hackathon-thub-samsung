import cv2
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras import optimizers
from keras.callbacks import Callback
# from keras import backend as K

def brg_to_rgb(bgr_image):
    b,g,r = cv2.split(bgr_image)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
    return rgb_img


def display_image(image, image_file):
    print(image_file)
    plt.imshow(brg_to_rgb(image)); plt.show(block=False); time.sleep(.2); plt.close()


def class_to_num(theClass):
    theClasses = ["A", "B", "C", "Five", "Point", "V"]
    return theClasses.index(theClass)


def num_to_class(num):
    theClasses = ["A", "B", "C", "Five", "Point", "V"]
    return theClasses[num]


def remove_outlier_images():
    # Remove images with height exceeding 200px,
    # because they need to be cropped to get the correct gesture image# Data
    # Data
    if data_type=="train":
        image_files = sorted(glob.glob(data_type + "/*/*.ppm"))
    elif data_type=="test":
        image_files = sorted(glob.glob(data_type + "/*/*/*.ppm"))
    # Remove
    for image_file in image_files:
        image = cv2.imread(image_file)/255.
        h, w, c = image.shape
        if h > 200:
            os.system("mv " + image_file + " to_crop/")


def check_shapes(image_files):
    shapes = []
    prev_len = 0
    for image_file in image_files:
        image = cv2.imread(image_file)/255.
        shapes.append(image.shape)
        if len(set(shapes)) > prev_len:
            prev_len += 1
            print(image_file)
            print(shapes[-1])


# Load all train csv data
def load_csv_data(csvs):
    data = np.empty((0, 2))
    for csv_file in csvs:
        # data = np.vstack((data, pd.read_csv(csv_file, encoding="ISO-8859-1").values))
        data = np.vstack((data, pd.read_csv(csv_file).values))
    # Return
    return data


# Write to csv
def write_to_csv(results, fname="results"):
    np.savetxt(fname+".csv", results, delimiter=',', header='FileName,Label', fmt='%s')


def make_data(data_type="train", normalize=False, ravel_x=True, one_hot_y=False):
    # Data
    if data_type=="train":
        image_files = sorted(glob.glob(data_type + "/*/*.ppm"))
    elif data_type=="test":
        image_files = sorted(glob.glob(data_type + "/*/*/*.ppm"))
    elif data_type=="val":
        image_files = sorted(glob.glob("cross_validation_data/*.ppm"))
    else:
        print("Wrong data type! train or test")
        return
    # # Check shapes
    # check_shapes(image_files)
    # Read images
    X = []
    Y = []
    if data_type=="val":
        val_data = load_csv_data(["lables_cross_validation_1.csv"])
        for i in range(len(val_data)):
            Y.append(class_to_num(val_data[i][1]))
        Y = np.array(Y)
    for image_file in image_files:
        if data_type=="train":
            Y.append(class_to_num(image_file.split('/')[-2]))
        elif data_type=="test":
            Y.append(class_to_num(image_file.split('/')[-3]))
        image = cv2.imread(image_file)
        im60 = cv2.resize(image, (60, 60))
        if normalize:
            im60 = im60 / 255.
        if ravel_x:
            X.append(im60.ravel())
        else:
            X.append(im60)
    # Y
    if one_hot_y:
        Y = np_utils.to_categorical(Y, num_classes=6)
    # Return
    return np.array(X), np.array(Y), image_files


def make_CNN(input_shape=(60, 60, 3)):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     kernel_regularizer=regularizers.l2(0.01),
                     input_shape=input_shape))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))
    return model


class PlotLossAndAcc(Callback):

    def on_train_begin(self, logs={}):
        # Initialization code
        self.epochs = 0
        self.trainLosses = []
        self.valLosses = []
        self.trainAccuracies = []
        self.valAccuracies = []

    def on_epoch_end(self, batch, logs={}):
        self.epochs += 1
        tl = logs.get('loss')
        ta = logs.get('acc')
        vl = logs.get('val_loss')
        va = logs.get('val_acc')
        self.trainLosses.append(tl)
        self.valLosses.append(vl)
        self.trainAccuracies.append(ta)
        self.valAccuracies.append(va)
        # Speaker-Independent
        # if self.epochs % X == 0:
        #     # Do stuff like printing metrics
        plt.close()
        self.plot_loss_and_acc()

    def plot_loss_and_acc(self):
        # summarize history for accuracy
        plt.subplot(121)
        plt.plot(self.trainAccuracies, label="train")
        plt.plot(self.valAccuracies, label="test")
        plt.title('accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        # plt.show()
        # summarize history for loss
        plt.subplot(122)
        plt.plot(self.trainLosses, label="train")
        plt.plot(self.valLosses, label="test")
        plt.title('loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        # plt.show(block=False)
        plt.savefig("a.png")


def gen_X_Y(orig_X, orig_Y, batch_size):
    X = np.array(orig_X)
    Y = np.array(orig_Y)
    full_idx = np.arange(len(X))
    while 1:
        np.random.shuffle(full_idx)
        X = X[full_idx]
        Y = Y[full_idx]
        for batch in range(len(X)//batch_size):
            X_batch = np.zeros((batch_size, 50, 50, 3))
            random_init_x = np.random.randint(low=0, high=10, size=(batch_size, 2))
            for i in range(batch_size):
                X_batch[i] = X[batch*batch_size:(batch+1)*batch_size][i][random_init_x[i][0]:random_init_x[i][0]+50, random_init_x[i][1]:random_init_x[i][1]+50]
            yield X_batch, Y[batch*batch_size:(batch+1)*batch_size]


