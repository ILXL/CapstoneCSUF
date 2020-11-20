import tensorflow.keras.layers as layers
import tensorflow as tf
import pandas as pd
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def RNNDiscriminator(dataset):
	"""Recurrent Neural Network

	Layout:
	  - Reshape Layer w/ Gaussian Noise Added
	  - 2 SimpleRNN TF layers
	  - Dense Layer with LeakyReLU activation
	  - Dense Layer with one node for output

	Args:
		dataset (Dataframe): Pandas Dataframe with columns and values
	
	Returns:
		TensorFlow Keras Sequential Neural Network
	"""
	features = [tf.compat.v2.feature_column.numeric_column(k, dtype=tf.dtypes.float64)
				for k in dataset.columns.values if (k != 'real' and k != 'actual')]
	
	print('Disc features:', len(features))
	model = tf.keras.models.Sequential()
	model.add(layers.Reshape([len(features), 1]))
	model.add(layers.GaussianNoise(1))
	model.add(layers.Dropout(.3))
	model.add(layers.SimpleRNN(128, return_sequences=True,
						  kernel_regularizer='l1', bias_regularizer='l2'))
	model.add(layers.SimpleRNN(128, return_sequences=False,
						  kernel_regularizer='l1', bias_regularizer='l2'))						  
	model.add(layers.Dropout(.3))
	model.add(layers.Dense(256,
						  kernel_regularizer='l1_l2'))
	model.add(layers.LeakyReLU(alpha=0.2))
	model.add(layers.Dense(1))
	return model


def generatorModelModified(dataset):
	"""Densely Connected Neural Network

	Layout:
	  - Dense Layer
	  - Dense Layer with Dropout and LeakyReLU activation
	  - Batch Normalization Layer
	  - Dense Layer with Dropout and LeakyReLU activation
	  - Batch Normalization Layer
	  - Dense Layer the same shape as dataset rows

	Args:
		dataset (Dataframe): Pandas Dataframe with columns and values
	
	Returns:
		TensorFlow Keras Dense Neural Network
	"""
	features = [tf.compat.v2.feature_column.numeric_column(k, dtype=tf.dtypes.float64)
				for k in dataset.columns.values if (k != 'real' and k != 'actual')]
	print('Gen features:', len(features))
	
	def customRELU(x):
		return tf.keras.activations.relu(x, max_value=100)

	model = tf.keras.models.Sequential()
	model.add(layers.Dense(256, input_shape=(len(features),),
							kernel_regularizer='l2', bias_regularizer='l1_l2'))
	model.add(layers.Dropout(.3))
	model.add(layers.Dense(512,
							kernel_regularizer='l1_l2', bias_regularizer='l1_l2'))
	model.add(layers.LeakyReLU(alpha=0.2))
	model.add(layers.BatchNormalization(momentum=0.8))
	model.add(layers.Dropout(0.4))
	model.add(layers.Dense(512,
							kernel_regularizer='l1_l2', bias_regularizer='l1_l2'))
	model.add(layers.LeakyReLU(alpha=0.2))
	model.add(layers.BatchNormalization(momentum=0.8))
	model.add(layers.Dense(len(features),
						   activation=customRELU))

	return model


##################################################
############# Unused Neural Networks #############
##################################################

def CNNModel(dataset):
	features = [tf.compat.v2.feature_column.numeric_column(k, dtype=tf.dtypes.float64)
				for k in dataset.columns.values if (k != 'real' and k != 'actual')]
	
	def customRELU(x):
		return tf.keras.activations.relu(x, max_value=100)

	model = tf.keras.models.Sequential()
	model.add(layers.Reshape([len(features), 1]))
	model.add(layers.Conv1D(filters=32,
                           kernel_size=(3,)
						   ))
	model.add(layers.MaxPooling1D())
	model.add(layers.Flatten())						   
	model.add(layers.Dense(128))
	model.add(layers.Dense(len(features),
						   activation=customRELU))
	return model


def generatorModel(dataset):
	features = [k for k in dataset.columns.values if k != 'real']

	model = tf.keras.models.Sequential()
	model.add(layers.Dense(256, input_shape=(len(features),)))
	model.add(layers.Dropout(.2))
	model.add(layers.Dense(256, activation="relu"))
	model.add(layers.Dense(len(features)))

	return model

def RNNGenerator(dataset):
	features = [tf.compat.v2.feature_column.numeric_column(k, dtype=tf.dtypes.float64)
				for k in dataset.columns.values if (k != 'real' and k != 'actual')]

	def customRELU(x):
		return tf.keras.activations.relu(x, max_value=100)

	model = tf.keras.models.Sequential()
	model.add(layers.Reshape([len(features), 1]))
	model.add(layers.SimpleRNN(128, return_sequences=True,
						  kernel_regularizer='l1', bias_regularizer='l2',
						  activation='relu'))
	model.add(layers.SimpleRNN(128, return_sequences=True,
						  kernel_regularizer='l1_l2', bias_regularizer='l2'))
	model.add(layers.SimpleRNN(128, return_sequences=False,
						  kernel_regularizer='l1_l2', bias_regularizer='l2'))						  
	model.add(layers.Dropout(.2))
	model.add(layers.Dense(128))
	model.add(layers.Dense(len(features),
						   activation=customRELU))

	return model


def discriminatorModel(dataset):

	features = [tf.compat.v2.feature_column.numeric_column(k, dtype=tf.dtypes.float64)
				for k in dataset.columns.values if (k != 'real' and k != 'actual')]

	model = tf.keras.models.Sequential()
	model.add(layers.Dense(256,
						   kernel_regularizer='l1',
						   input_shape=(len(features),)))
	model.add(layers.Dropout(.2))
	model.add(layers.Dense(256, kernel_regularizer='l2'))
	model.add(layers.Dense(1))
	return model



if __name__ == "__main__":
	print('Can only be used as neural network module')
else:
	print('Successfully loaded', __name__)
