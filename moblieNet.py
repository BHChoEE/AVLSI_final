import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

import sys
import pandas as pd
import numpy as np
import pickle

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split

from keras import backend as K

if __name__ == '__main__':

	K.clear_session()

	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.25
	set_session(tf.Session(config=config))

	batch_size = 64
	epochs = 150 ## we use automatic early stopping

	train = pd.read_csv(sys.argv[1], sep=',| ', header=None, skiprows=1, engine='python')
	train = np.array(train)

	X_train, X_val, y_train, y_val = train_test_split(train[:, 1:], train[:, 0], random_state=0, test_size=0.1)

	num_classes = np.amax(train[:, 0]) + 1 # label starts from 0

	# X_train = X_train / 255
	# X_val = X_val / 255
	X_train = X_train.reshape((-1, 48, 48, 1))
	X_val = X_val.reshape((-1, 48, 48, 1))
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_val = keras.utils.to_categorical(y_val, num_classes)

#### build model: CNN
	model = keras.application.mobilenet.MobileNet(input_shape=(48, 48, 1), weights='None', classes=num_classes)

#### compile model
	adam_01 = keras.optimizers.Adam(lr=0.01)
	model.compile(loss='categorical_crossentropy', optimizer=adam_01, metrics=['accuracy'])

	print('compile model')

#### data augmentation
	datagen = ImageDataGenerator(
		rescale=1./255,
	    featurewise_center=False,
	    featurewise_std_normalization=False,
	    rotation_range=20,
	    width_shift_range=0.2,
	    height_shift_range=0.2,
	    horizontal_flip=True)

	val_datagen = ImageDataGenerator(rescale=1./255)

	# compute quantities required for featurewise normalization
	# (std, mean, and principal components if ZCA whitening is applied)
	datagen.fit(X_train)

	print('data augmentation')

	# fits the model on batches with real-time data augmentation:
	early_stopping_callback = EarlyStopping(monitor='val_loss', patience=50)
	checkpoint_callback = ModelCheckpoint(sys.argv[2]+'.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True),
            	steps_per_epoch=len(X_train) / batch_size, validation_data=val_datagen.flow(X_val, y_val, batch_size=batch_size, shuffle=True),
            	validation_steps=len(X_train) / batch_size,epochs=epochs, callbacks=[early_stopping_callback, checkpoint_callback])

	pickle.dump(history, open('history.pkl', 'wb'))
	# model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
	#                     steps_per_epoch=len(X_train) / batch_size, epochs=epochs,
	#                     validation_data=val_datagen.flow(X_val, y_val, batch_size=batch_size),
	#                     validation_steps=100)

	print('model fit generator')

	# here's a more "manual" example
	# for e in range(epochs):
	#     print('Epoch', e)
	#     batches = 0
	#     for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=32):
	#         model.fit(X_batch, y_batch)
	#         batches += 1
	#         if batches >= len(X_train) / 32:
	#             # we need to break the loop by hand because
	#             # the generator loops indefinitely
	#             break

#### fit model
	# model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, shuffle=True)
	# earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
	# model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[earlyStopping], validation_split=0.15, validation_data=None, shuffle=True)

#### save model
	# model.save(sys.argv[2])

#### evaluate the model
	scores = model.evaluate(X_val, y_val, batch_size=batch_size)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#### predict
	test = pd.read_csv(sys.argv[3], sep=',| ', header=None, skiprows=1, engine='python')
	test = np.array(test)
	X_test = (test[:, 1:]) / 255
	X_test = X_test.reshape((-1, 48, 48, 1))

	predict = model.predict_classes(X_test, batch_size=batch_size)

	df = pd.DataFrame(predict, columns=['label'])
	df.index.name = 'id'
	df.to_csv(sys.argv[4], encoding='utf-8')	