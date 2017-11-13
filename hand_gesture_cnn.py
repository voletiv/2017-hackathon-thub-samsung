import sys
import yaml

del sys.modules['hand_gesture_functions']
from hand_gesture_functions import *

# # Make train data
# # n x w x w x 3
# train_X, train_Y = make_data("train", normalize=True, one_hot_y=True)

# # Make test data
# # t x w x w x 3
# test_X, test_Y = make_data("test", normalize=True, one_hot_y=True)

# Make CNN
model = make_CNN(input_shape=(50, 50, 3))
model.summary()

# Compile
model.compile(loss="categorical_crossentropy",
    optimizer=optimizers.RMSprop(),
    metrics=['accuracy'])

# batch size
batch_size = 64
train_steps_per_epoch = len(train_X)//batch_size
test_steps_per_epoch = len(test_X)//batch_size

# Train gen
train_gen = gen_X_Y(train_X, train_Y, batch_size)
test_gen = gen_X_Y(test_X, test_Y, batch_size)

# Callback
plot_loss_and_acc = PlotLossAndAcc()

# Train CNN
n_epochs = 100
history = model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=n_epochs,
    validation_data=test_gen, validation_steps=test_steps_per_epoch,
    callbacks=[plot_loss_and_acc])

yaml_string = model.to_yaml()
with open("model.yml", 'w') as outfile:
    yaml.dump(yaml_string, outfile, default_flow_style=False)

model.save_weights('cnn_model.hdf5')

# Save predictions as csv


# l, a = model.evaluate_generator(test_gen, test_steps_per_epoch)
