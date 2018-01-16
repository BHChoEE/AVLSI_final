import numpy as np
import pandas as pd

import pickle
import argparse

def preprocessing(filePath):
    from keras.preprocessing import image as image_utils
    pathDict = pd.read_csv(filePath, header = None).as_matrix()
    train_data = np.zeros((100000, 64, 64, 3))
    ## for test running
    #train_data = np.zeros((1000, 64, 64, 3))
    imageDict = []
    for index, fileNames in enumerate(pathDict):
        fileName = fileNames[0]
        dataPath = './train/' + fileName + '/images/' + fileName + '_'
        imageSet = np.zeros((500, 64, 64, 3))
        for imgNum in range(500):
        ### for test running
        #imageSet = np.zeros((5, 64, 64, 3))
        #for imgNum in range(5):
            imgPath = dataPath + str(imgNum) + '.JPEG'
            image = image_utils.load_img(imgPath, target_size = (64, 64))
            image = image_utils.img_to_array(image)
            imageSet[imgNum] = image
            train_data[index * 500 + imgNum] = image
            ### for test running
            #train_data[index * 5 + imgNum] = image
        imageDict.append(imageSet)
    
    print('Read Done!!')

    train_label = np.repeat(np.arange(200), 500)
    #train_label = np.repeat(np.arange(200), 5)
    '''
    for index, imageSet in enumerate(imageDict):
        with open('./pkl/class' + str(index) + '.pkl', 'wb') as ofs:
            pickle.dump(imageSet, ofs)
    #with open(outPath, 'wb') as fp:
    #    pickle.dump(imageDict, fp)
    '''     
    return train_data, train_label   



def MobileNet(input_tensor=None, input_shape=None, alpha=1, shallow=False, classes=1000):
    from keras.applications.imagenet_utils import _obtain_input_shape
    from keras import backend as K
    from keras.layers import Input, Convolution2D, \
        GlobalAveragePooling2D, Dense, BatchNormalization, Activation
    from keras.models import Model
    from keras.engine.topology import get_source_inputs
    from depthwise_conv2d import DepthwiseConvolution2D

    """Instantiates the MobileNet.Network has two hyper-parameters
        which are the width of network (controlled by alpha)
        and input size.
        
        # Arguments
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(224, 224, 3)` (with `channels_last` data format)
                or `(3, 224, 244)` (with `channels_first` data format).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 96.
                E.g. `(200, 200, 3)` would be one valid value.
            alpha: optional parameter of the network to change the 
                width of model.
            shallow: optional parameter for making network smaller.
            classes: optional number of classes to classify images
                into.
        # Returns
            A Keras model instance.
        """
    '''
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=96,
                                      data_format=K.image_data_format(),)
                                      #include_top=True)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    '''
    img_input = Input(shape = (64, 64, 3))
    
    x = Convolution2D(int(32 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = DepthwiseConvolution2D(int(32 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(64 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = DepthwiseConvolution2D(int(64 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConvolution2D(int(128 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConvolution2D(int(128 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConvolution2D(int(256 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = DepthwiseConvolution2D(int(256 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    if not shallow:
        for _ in range(5):
            x = DepthwiseConvolution2D(int(512 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Convolution2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

    x = DepthwiseConvolution2D(int(512 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(1024 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = DepthwiseConvolution2D(int(1024 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(1024 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = GlobalAveragePooling2D()(x)
    out = Dense(classes, activation='softmax')(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, out, name='mobilenet')
    model.summary()
    return model

def train(train_data, train_label, model, n_epoch, batch_size):
    from keras import optimizers
    from keras import callbacks
    from sklearn.model_selection import train_test_split
    X_train, val_train, X_label, val_label = train_test_split(train_data, train_label, test_size = 0.1, shuffle = True) 
    rmsprop = optimizers.RMSprop(lr = 0.008)
    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = rmsprop,
        metrics = ['mae', 'acc']
    )
    earlystopping = callbacks.EarlyStopping(
        monitor = 'val_acc',
        patience = 20,
        verbose = 1,
        mode = 'max'
    )
    checkpoint = callbacks.ModelCheckpoint(
        filepath = './models/best.hdf5',
        verbose = 1,
        save_best_only = True,
        save_weights_only = True,
        monitor = 'val_acc',
        mode = 'max'
    )
    hist = model.fit(
        X_train, X_label,
        validation_data = (val_train, val_label),
        epochs = n_epoch,
        batch_size = batch_size,
        callbacks = [earlystopping, checkpoint]
    )
    model.save('testModel.h5')
    return model


def main():
    ### data preprocessing
    parser = argparse.ArgumentParser(description = "MobileNet on Tiny ImageNet Dataset")
    # Retrievement argument
    parser.add_argument('-f', '--file', type = str, help = 'FileDict Name Input', default = "")
    parser.add_argument('-e', '--epoch', type = int, help = 'number of epochs', default = 1000)
    parser.add_argument('-b', '--batch_size', type = int, help = 'number of batch size', default = 128)
    args = parser.parse_args()

    train_data, train_label = preprocessing(args.file)
    from keras.utils import to_categorical
    train_label = to_categorical(train_label, num_classes = 200)
    ### model   
    model = MobileNet(input_shape = (64, 64, 3), shallow = True, classes = 200)
    ### training
    trained_model = train(train_data, train_label, model, args.epoch, args.batch_size)

if __name__ == '__main__':
    main()

    
