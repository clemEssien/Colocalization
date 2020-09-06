import os
import glob
import keras
import generator as gen
import utils as util
#from keras_video import VideoFrameGenerator

# for data augmentation
GLOB_PATTERN ='dataset/{classname}/*/*.png'
# use sub directories names as classes
classes = [i.split(os.path.sep)[1] for i in glob.glob('dataset/*')]
classes.sort()
# some global params
SIZE = (30, 30)
CHANNELS = 1
NBFRAME = 10
BS = 4

data_aug = keras.preprocessing.image.ImageDataGenerator(
    zoom_range=.1,
    horizontal_flip=True,
    rotation_range=8,
    width_shift_range=.2,
    height_shift_range=.2)

train = gen.VideoFrameGenerator(
    nbframe=NBFRAME,
    classes=classes,
    batch_size=BS,
    use_frame_cache=False,
    target_shape=SIZE,
    shuffle=True,
    transformation=data_aug,
    split=.20,
    nb_channel=CHANNELS,
    glob_pattern=GLOB_PATTERN,
    _validation_data=None
)

valid = train.get_validation_generator()
import keras_video.utils
# util.show_sample(train)

from keras.layers import Conv2D, BatchNormalization, \
    MaxPool2D, GlobalMaxPool2D
def build_convnet(shape=(30, 30, 1)):
    momentum = .9
    model = keras.Sequential()
    model.add(Conv2D(4, (3,3), input_shape=shape,
        padding='same', activation='relu'))
    model.add(Conv2D(4, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    model.add(MaxPool2D())
    
    model.add(Conv2D(8, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(8, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    model.add(MaxPool2D())
    
    model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    # model.add(MaxPool2D())
    
    # model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
    # model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    # flatten...
    model.add(GlobalMaxPool2D())
    return model


from keras.layers import TimeDistributed, GRU, Dense, Dropout
def action_model(shape=(10, 30, 30, 1), nbout=3):
    # Create our convnet with (112, 112, 3) input shape
    convnet = build_convnet(shape[1:])
    
    # then create our final model
    model = keras.Sequential()
    # add the convnet with (5, 112, 112, 3) shape
    model.add(TimeDistributed(convnet, input_shape=shape))
    # here, you can also use GRU or LSTM
    model.add(GRU(64))
    # and finally, we make a decision network
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(nbout, activation='softmax'))
    return model


INSHAPE=(NBFRAME,) + SIZE + (CHANNELS,) # (5, 112, 112, 3)
model = action_model(INSHAPE, len(classes))
optimizer = keras.optimizers.Adam(0.001)
model.compile(
    optimizer,
    'categorical_crossentropy',
    metrics=['acc']
)

EPOCHS=50
# create a "chkp" directory before to run that
# because ModelCheckpoint will write models inside
callbacks = [
    keras.callbacks.ReduceLROnPlateau(verbose=1),
    keras.callbacks.ModelCheckpoint(
        'chkp/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        verbose=1),
]
model.fit_generator(
    train,
    validation_data=valid,
    verbose=1,
    epochs=EPOCHS,
    callbacks=callbacks
)

# model.fit_generator(
#     train_generator,
#     steps_per_epoch=2000,
#     epochs=50,
#     validation_data=validation_generator,
#     validation_steps=800)


# use sub directories names as classes
# classes = [i.split(os.path.sep)[1] for i in glob.glob('dataset/*')]
# classes.sort()
# # some global params
# SIZE = (30, 30)
# CHANNELS = 3
# NBFRAME = 10
# BS = 4
# # pattern to get videos and classes
# glob_pattern='dataset/{classname}/*/*.png'

# # for data augmentation
# data_aug = keras.preprocessing.image.ImageDataGenerator(
#     zoom_range=.1,
#     horizontal_flip=True,
#     rotation_range=8,
#     width_shift_range=.2,
#     height_shift_range=.2)
# # Create video frame generator
# train = VideoFrameGenerator(
    # classes=classes, 
    # glob_pattern=glob_pattern,
    # nb_frames=NBFRAME,
    # split_val=.20, 
    # shuffle=True,
    # batch_size=BS,
    # target_shape=SIZE,
    # nb_channel=CHANNELS,
    # transformation=data_aug,
    # use_frame_cache=True)

# valid = train.get_validation_generator()

# import keras_video.utils
# keras_video.utils.show_sample(train)

# print((train.shuffle))


'''Model one: a CNN that looks at single images.
Model two: a RNN that at the sequences of the output of the CNN from model one.

So for example the CNN should see 5 images and this sequence of 5 outputs from the CNN should be passed on to the RNN.

The input data is in the following format:
(number_of_images, width, height, channels) = (4000, 120, 60, 1)'''
# cnn = Sequential()
# cnn.add(Conv2D(16, (50, 50), input_shape=(120, 60, 1)))

# cnn.add(Conv2D(16, (40, 40)))

# cnn.add(Flatten()) #

# rnn = Sequential()

# rnn = GRU(64, return_sequences=False, input_shape=(120, 60))
# dense = Sequential()
# dense.add(Dense(128))
# dense.add(Dense(64))

# dense.add(Dense(1)) # Model output

# main_input = Input(shape=(5, 120, 60, 1)) # Data has been reshaped to (800, 5, 120, 60, 1)

# model = TimeDistributed(cnn)(main_input) # this should make the cnn 'run' 5 times?
# model = rnn(model) # combine timedistributed cnn with rnn
# model = dense(model) # add dense
# final_model = Model(inputs=main_input, outputs=model)

# final_model.compile...
# final_model.fit...