import sys
import pandas as pd
import numpy as np
import pickle
import argparse

'keras API'
from keras import utils
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

'model training'
def train(trainDataPath, iteration, modelName):
	'parameter'
	batch_size = 64
	epochs = iteration ## we use automatic early stopping

	'parsing'
	print('start parsing............')
	train = pd.read_csv(trainDataPath, sep=',| ', header=None, skiprows=1, engine='python')
	train = np.array(train)

	X_train, X_val, y_train, y_val = train_test_split(train[:, 1:], train[:, 0], random_state=0, test_size=0.1)

	num_classes = np.amax(train[:, 0]) + 1 # label starts from 0

	X_train = X_train.reshape((-1, 48, 48, 1))
	X_val = X_val.reshape((-1, 48, 48, 1))
	Y_train = utils.to_categorical(y_train, num_classes)
	Y_val = utils.to_categorical(y_val, num_classes)
	#print(X_train.shape)
	#print(X_val.shape)
	#print(Y_train.shape)
	#print(Y_val.shape)
	'model training'
	print('start training model.........')
	## build model: mobileNet
	model = MobileNet(
                        input_shape = (48, 48, 1),
                        alpha = 0.1,
                        dropout = 1e-3,
                        include_top = True,
                        weights = None,
                        input_tensor = None,
                        pooling = None,
                        classes = num_classes)
	## compile model
	adam = Adam(lr=0.01)
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	model.summary()

	## add noise
	datagen = ImageDataGenerator(
		rescale = 1./255,
	    featurewise_center = False,
	    featurewise_std_normalization = False,
	    rotation_range = 20,
	    width_shift_range = 0.2,
	    height_shift_range = 0.2,
	    horizontal_flip=True)
	val_datagen = ImageDataGenerator(rescale=1./255)

	## data augmentation
	datagen.fit(X_train)
	print('data augmentation done')
		
	# fits the model on batches with real-time data augmentation:
	early_stopping = EarlyStopping(monitor='val_acc', patience= 10)
	checkpoint = ModelCheckpoint('./bin/best.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	history = model.fit_generator(
				datagen.flow(X_train, Y_train, batch_size = batch_size, shuffle = True),
            	                steps_per_epoch = len(X_train) / batch_size,
				validation_data = val_datagen.flow(X_val, Y_val, batch_size = batch_size, shuffle = True),
            	                validation_steps = len(X_train) / batch_size, 
                                epochs = epochs, 
				callbacks = [early_stopping, checkpoint]
				)
	print('model fit generator done')

	model.save("./bin/" + modelName + '.h5')

'model testing'
def test(testDataPath, modelName, outputFileName):
	test = pd.read_csv(testDataPath, sep=',| ', header=None, skiprows=1, engine='python')
	test = np.array(test)
	X_test = (test[:, 1:]) / 255
	X_test = X_test.reshape((-1, 48, 48, 1))

	## load model 
	model = load_model("./bin/" + modelName + '.h5')

	#### evaluate the model
	scores = model.evaluate(X_val, y_val, batch_size=batch_size)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

	#### predict
	predict = model.predict_classes(X_test, batch_size=batch_size)
	df = pd.DataFrame(predict, columns=['label'])
	df.index.name = 'id'
	df.to_csv(outputFileName, encoding='utf-8')

def main():
	'for station training'
	K.clear_session()
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.25
	set_session(tf.Session(config=config))
	
	'argument processing'
	parser = argparse.ArgumentParser()
	parser.add_argument('-train', '--train', type = str, help = 'define task be training, and get trainingData input path')
	parser.add_argument('-test', '--test', type = str, help = 'define task is testing, and get testingData input path')
	parser.add_argument('-i', '--iter', type = int, help = 'if training, give iteration number')
	parser.add_argument('-m', '--model', type = str, help = 'modelName path')
	parser.add_argument('-o', '--outputFileName', type = str, help = 'output File Name')
	args = parser.parse_args()
	if args.train:
		if args.iter != None:
			train(args.train, args.iter, args.model)
		else: # default iteration number = 20;
			train(args.train, 20, args.model)
	if args.test:
		test(args.test, args.model, args.outputFileName)

if __name__ == '__main__':
	main()

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
