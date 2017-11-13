from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import resource
np.random.seed(1337)  # for reproducibility

# from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from hand_gesture_functions import *
from resnet import *


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def make_pairs(X, y, n_classes=6):
    digit_indices = [np.where(y == i)[0] for i in range(n_classes)]
    train_pairs, train_y = create_pairs(X, digit_indices)
    return train_pairs, train_y


def create_pairs(x, digit_indices):
    """ Positive and negative pair creation.
        Alternates between positive and negative pairs.
    """
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(len(digit_indices))]) - 1
    for d in range(len(digit_indices)):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            pairs += [[x[z1], x[z2]]]
            inc = np.random.randint(1, len(digit_indices))
            dn = (d + inc) % len(digit_indices)
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    pairs = np.array(pairs)
    labels = np.array(labels)
    return pairs, labels


def dense_base_network(input_shape):
    """ Base network to be shared (eq. to feature extraction).
    """
    seq = Sequential()
    seq.add(Dense(128, input_shape=(input_shape), activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    return seq


def make_siamese_model(input_shape, network_type="cnn"):
    # Model
    # network definition
    network_types = ["dense", "cnn", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    assert (network_type in network_types), "network_type must be one of "+str(network_types)

    if network_type=="dense":
        assert (len(input_shape) == 1),"Input shape must have 1 dimension!"
        base_network = dense_base_network(input_shape)
    else:
        assert (len(input_shape) == 3),"Input shape must have 3 dimensions!"
        if network_type=="cnn":
            base_network = make_CNN(input_shape)
        elif network_type=="resnet18":
            base_network = ResnetBuilder.build_resnet_18(input_shape=input_shape, num_outputs=6)
        elif network_type=="resnet34":
            base_network = ResnetBuilder.build_resnet_34(input_shape=input_shape, num_outputs=6)
        elif network_type=="resnet50":
            base_network = ResnetBuilder.build_resnet_50(input_shape=input_shape, num_outputs=6)
        elif network_type=="resnet101":
            base_network = ResnetBuilder.build_resnet_101(input_shape=input_shape, num_outputs=6)
        elif network_type=="resnet152":
            base_network = ResnetBuilder.build_resnet_152(input_shape=input_shape, num_outputs=6)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)
    return model

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def my_image_gen():
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10.,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        zoom_range = 0.2)
    return datagen


def gen_shuffled_augmented_train_pairs_data(X_train_orig, y_train_orig, batch_size):
    X_train = np.array(X_train_orig)
    y_train = np.array(y_train_orig)
    full_idx = np.arange(len(y_train))
    datagen = my_image_gen()
    while 1:
        # print("GEN START: Memory usage: %s (kb)" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        np.random.shuffle(full_idx)
        X_train = X_train[full_idx]
        y_train = y_train[full_idx]
        # print("GEN MID: Memory usage: %s (kb)" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        X_train_pairs, y_train_pairs = make_pairs(X_train, y_train)
        # print("GEN AFTER MAKING PAIRS: Memory usage: %s (kb)" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # 0
        gen_0 = datagen.flow(X_train_pairs[:, 0], y_train_pairs, batch_size=batch_size, shuffle=False)
        # 1
        gen_1 = datagen.flow(X_train_pairs[:, 1], y_train_pairs, batch_size=batch_size, shuffle=False)
        for batch in range(len(y_train_pairs)//batch_size):
            X_0, _ = gen_0.next()
            # print("GEN AFTER GEN_0: Memory usage: %s (kb)" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            X_1, y = gen_1.next()
            # print("GEN AFTER GEN_1: Memory usage: %s (kb)" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            yield [X_0, X_1], y
        # del X_train_pairs, y_train_pairs
        # print("GEN AFTER DEL: Memory usage: %s (kb)" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


class PlotLossAndAccSiamese(Callback):

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
        ta = logs.get('accuracy')
        vl = logs.get('val_loss')
        va = logs.get('val_accuracy')
        self.trainLosses.append(tl)
        self.valLosses.append(vl)
        self.trainAccuracies.append(ta)
        self.valAccuracies.append(va)
        # Speaker-Independent
        # if self.epochs % X == 0:
        #     # Do stuff like printing metrics
        # print('[PLOT]: Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
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

