#################################################
# RESNET
#################################################

import tqdm

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from hand_gesture_functions import *

from resnet import *

checkpointer = ModelCheckpoint(filepath='resnet101_weights.hdf5', verbose=1, save_best_only=True)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
# early_stopper = EarlyStopping(min_delta=0.001, patience=10)
plot_loss_and_calc =  PlotLossAndAcc()

batch_size = 32
nb_classes = 6
n_epochs = 200
data_augmentation = True

input_shape = (60, 60, 3)
num_outputs = 6

X_train, Y_train = make_data("train", normalize=False, ravel_x=False, one_hot_y=True)
X_test, Y_test = make_data("test", normalize=False, ravel_x=False, one_hot_y=True)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# subtract mean and normalize
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
X_train /= 128.
X_test /= 128.

# RESNET-18
# # basic_block
# model = ResnetBuilder.build_resnet_18(input_shape=input_shape, num_outputs=num_outputs)
# # bottleneck
# model = ResnetBuilder.build(input_shape, num_outputs, bottleneck, [2, 1, 1, 1])

# RESNET-34
model = ResnetBuilder.build_resnet_101(input_shape=input_shape, num_outputs=num_outputs)

model.compile(loss='categorical_crossentropy',
              # optimizer='adam',
              optimizer='adam',
              metrics=['accuracy'])

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=n_epochs,
              validation_data=(X_test, Y_test),
              shuffle=True,
              callbacks=[lr_reducer, early_stopper, csv_logger])
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True),
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        max_q_size=100, verbose=1,
                        validation_data=(X_test, Y_test), epochs=n_epochs,
                        callbacks=[checkpointer, lr_reducer, plot_loss_and_calc])

'''

##########################################################
# TEST
##########################################################

y_test_preds = []
test_on = 400
for x_test in tqdm.tqdm(X_test):
    y_preds = []
    for theClass in range(6):
        X_for_test_pred = [ np.array([x_test]*test_on), X_train[y_train==theClass][np.random.choice(np.arange(np.sum(y_train==theClass)), test_on, replace=False)] ]
        y_preds.append(np.mean(model.predict(X_for_test_pred)))
        # print(y_preds)
    y_test_preds.append(np.argmin(y_preds))


##########################################################
# VAL
##########################################################

y_test_preds = []

X_val, y_val, file_names = make_data("val", normalize=normalize, ravel_x=ravel_x, one_hot_y=one_hot_y)

file_numbers = [int(f.split('/')[-1].split('.')[0]) for f in file_names]
sorted_order = np.argsort(file_numbers)
X_val = X_val[sorted_order]
y_val = y_val[sorted_order]
file_names = np.array(file_names)[sorted_order]

y_val_preds = []
test_on = 400
for x_val in tqdm.tqdm(X_val):
    y_preds = []
    for theClass in range(6):
        X_for_val_pred = [ np.array([x_val]*test_on), X_train[y_train==theClass][:test_on] ]
        y_preds.append(np.mean(model.predict(X_for_val_pred)))
        # print(y_preds)
    y_val_preds.append(np.argmin(y_preds))

results = []
for i in range(len(y_val_preds)):
    results.append([file_names[i].split('/')[-1], num_to_class(y_val_preds[i])])

write_to_csv(results)

'''
