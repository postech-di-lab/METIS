# # baseline model with dropout on the cifar10 dataset
# import sys
# from matplotlib import pyplot
# from keras.datasets import cifar10
# from keras.utils import to_categorical
# from keras.models import Sequential
# from keras.layers import Conv2D
# from keras.layers import MaxPooling2D
# from keras.layers import Dense
# from keras.layers import Flatten
# from keras.layers import Dropout
# from keras.optimizers import SGD

# # load train and test dataset
# def load_dataset():
#     # load dataset
#     (trainX, trainY), (testX, testY) = cifar10.load_data()
#     # one hot encode target values
#     trainY = to_categorical(trainY)
#     testY = to_categorical(testY)
#     return trainX, trainY, testX, testY

# # scale pixels
# def prep_pixels(train, test):
#     # convert from integers to floats
#     train_norm = train.astype('float32')
#     test_norm = test.astype('float32')
#     # normalize to range 0-1
#     train_norm = train_norm / 255.0
#     test_norm = test_norm / 255.0
#     # return normalized images
#     return train_norm, test_norm

# # define cnn model
# def define_model():
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
#     model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Dropout(0.2))
#     model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
#     model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Dropout(0.2))
#     model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
#     model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Dropout(0.2))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
#     model.add(Dropout(0.2))
#     model.add(Dense(10, activation='softmax'))
#     # compile model
#     opt = SGD(lr=0.001, momentum=0.9)
#     model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# # plot diagnostic learning curves
# def summarize_diagnostics(history):
#     # plot loss
#     pyplot.subplot(211)
#     pyplot.title('Cross Entropy Loss')
#     pyplot.plot(history.history['loss'], color='blue', label='train')
#     pyplot.plot(history.history['val_loss'], color='orange', label='test')
#     # plot accuracy
#     pyplot.subplot(212)
#     pyplot.title('Classification Accuracy')
#     pyplot.plot(history.history['acc'], color='blue', label='train')
#     pyplot.plot(history.history['val_acc'], color='orange', label='test')
#     # save plot to file
#     filename = sys.argv[0].split('/')[-1]
#     pyplot.savefig(filename + '_plot.png')
#     pyplot.close()

# # run the test harness for evaluating a model
# def run_test_harness():
#     # load dataset
#     trainX, trainY, testX, testY = load_dataset()
#     # prepare pixel data
#     trainX, testX = prep_pixels(trainX, testX)
#     # define model
#     model = define_model()
#     # fit model
#     history = model.fit(trainX, trainY, epochs=100, batch_size=64, validation_data=(testX, testY), verbose=1)
#     # evaluate model
#     _, acc = model.evaluate(testX, testY, verbose=1)
#     print('> %.3f' % (acc * 100.0))
#     # learning curves
#     summarize_diagnostics(history)

# # entry point, run the test harness
# run_test_harness()

#############################################################################################

from __future__ import print_function

import os
os.environ['PYTHONHASHSEED'] = '0'

import keras
import random as rn
rn.seed(0)

import numpy as np
np.random.seed(0)


import  tensorflow as tf

tf.set_random_seed(0)

from keras import backend as K
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Activation, Flatten, Dropout, BatchNormalization, Permute
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import PReLU, LeakyReLU
from keras.datasets import cifar10, mnist, cifar100
from keras import regularizers
from keras.callbacks import LearningRateScheduler

from keras.optimizers import rmsprop, Adam, SGD
from keras.engine.topology import Layer
from keras import initializers
from keras import activations

from LearningRateMultiplier import LearningRateMultiplier
#from LearningRateMultiplier import DecayMultiplier

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn

from keras.callbacks import LambdaCallback

import sys
sys.setrecursionlimit(10000)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config = config)
K.tensorflow_backend.set_session(sess)

#import keras
#training
batch_size = 64#x_train.shape[0]

class myDense(Layer):
    def __init__(self, units, exp=False, activation=None, kernel_initializer='glorot_uniform', name=None, trainable=True, norm=False, **kwargs):
        super(myDense, self).__init__(name=name, trainable=trainable, **kwargs)
        self.units = int(units)
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        # self.norm = norm
        self.exp = exp
        self.dtype = tf.float32

    def build(self, input_shape):
        last_dim = input_shape[-1]
        self.kernel = self.add_weight(
            'kernel',
            shape=[self.units, last_dim],
            initializer=self.kernel_initializer,
            trainable=True)

        self.bias = self.add_weight(
            'bias',
            shape=[self.units,],
            initializer=initializers.get('zeros'),
            trainable=True)

        if self.exp:
            self.e = self.add_weight(
                'e',
                shape=[self.units, last_dim],
                initializer=initializers.get('ones'),
                trainable=True)

        self.built = True

    def call(self, inputs):
        expanded_inputs = K.repeat(inputs, self.units) # None * 10 * 2048

        if self.exp:
            # bias = 0.2
            # exp_input = K.pow(expanded_inputs + bias, self.e)
            # x_sign = K.cast(K.sign(expanded_inputs), tf.float32)
            x_pos = K.relu(expanded_inputs)
            x_neg = -K.relu(-expanded_inputs)
            exp_input_pos = (K.pow(x_pos, self.e))
            exp_input = exp_input_pos + x_neg


            input_mult = exp_input * self.kernel
        else:
            input_mult = expanded_inputs * self.kernel

        outputs = K.sum(input_mult, axis=2)

        final_outputs = K.bias_add(outputs, self.bias)

        return self.activation(final_outputs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

class BN(Layer):
    def __init__(self, trainable=True, **kwargs):
        super(BN, self).__init__(trainable=trainable, **kwargs)

    def build(self, input_shape):
        last_dim = input_shape[-1]
        self.kernel = self.add_weight(
            'kernel',
            shape=[last_dim],
            initializer=initializers.get('ones'),
            trainable=True)

        self.bias = self.add_weight(
            'bias',
            shape=[last_dim],
            initializer=initializers.get('zeros'),
            trainable=True)

    def call(self, x):
        # JUST BATCH NORMALIZATION
        x_mean = K.mean(x, axis=0)
        x_var = K.var(x, axis=0)
        epsilon = 0.00
        x_norm = (x - x_mean) / K.sqrt(x_var + epsilon)
        x_final = K.bias_add(x_norm * self.kernel, self.bias)

        return x_final

    def compute_output_shape(self, input_shape):
        return input_shape
       




class myBatchNormalization(Layer):
    def __init__(self, units, gamma_initializer='ones', beta_initializer='zeros', name=None, trainable=True, **kwargs):
        super(myBatchNormalization, self).__init__(name=name, trainable=trainable, **kwargs)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_initializer = initializers.get(beta_initializer)

        self.dtype = tf.float32
        self.units = units

    def build(self, input_shape):
        input_shape = input_shape[0]
        last_dim = input_shape[-1]
        self.gamma = self.add_weight(
            'gamma',
            shape=[last_dim,],
            initializer=self.gamma_initializer,
            trainable=True)

        self.beta = self.add_weight(
            'beta',
            shape=[last_dim,],
            initializer=self.beta_initializer,
            trainable=True)

    def call(self, x):
        x1 = x[0]
        x_mean = K.mean(x1, axis=0)
        x_var = K.var(x1, axis=0)
        epsilon = 0.001
        x_norm = (x1 - x_mean) / K.sqrt(x_var + epsilon)
        x_mult = K.bias_add(x_norm * self.gamma, self.beta)
        x_final = x_mult

        # gradient
        b_size = K.cast(K.shape(x1)[0], tf.float32)
        E = x_mean
        G = K.mean(K.pow(x1, 2), axis=0)
        GE2 = G - K.pow(E, 2)
        grad = self.gamma * K.pow(GE2, -3.0/2.0) * ((1.0 - 1 / b_size) * GE2 - (1 / b_size) * K.pow(x1 - E, 2))

        return [x_final, x[1], grad]

    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1], input_shape[0][-1]]

class ExtendedRelu(Layer):
    def __init__(self, norm, trainable=True, name=None, **kwargs):
        super(ExtendedRelu, self).__init__(name=name, trainable=trainable, **kwargs)
        self.norm = norm

    def build(self, inshape):
        inshape = inshape[0]
        self.normalize2 = self.add_weight(name="normalize2", shape=(inshape[1:]), initializer=initializers.Constant(0.1))
        self.incline = self.add_weight(name="incline", shape=(inshape[1:]), initializer=initializers.Constant(0.1))

    def call(self, x):
        x0 = x[0]
        x1 = x[1]

        num_neg = K.sum(K.cast(x0 < 0.0, tf.float32), axis=-1)
        num_x = K.cast(K.shape(x0)[1], tf.float32)
        # print num_neg

        # print num_x
        # input()

        # incline = K.reshape(K.permute_dimensions(1.0 / K.pow(K.permute_dimensions(K.repeat(K.reshape(K.sum(x1, axis=0) * 10, shape=[1, -1]), K.shape(x0)[0]), (0, 2, 1)), num_x / num_neg), (2, 1, 0)), shape=[K.shape(x0)[0], -1]) # * x[2]) # (3072 * 10, 10)
        #incline = K.reshape(K.permute_dimensions(1.0 / K.pow(K.permute_dimensions(K.repeat(K.reshape(K.sum(x1, axis=0), shape=[1, -1]), K.shape(x0)[0]), (0, 2, 1)) * num_neg, 2.0), (2, 1, 0)), shape=[K.shape(x0)[0], -1]) # * x[2]) # (3072 * 10, 10)

        # K.repeat(inputs, self.units)
        

        t = 1000.0
        # neg = K.cast(x0 < 0.0, tf.float32)
        x_neg = -K.relu(-x0)
        x_pos = K.relu(x0)
        x_exp_pos = 1.0 * (x_pos - K.round(t * x_pos) / t) + K.round(t * x_pos) / t

        x_exp_neg = self.incline * (x_neg - K.round(t * x_neg + 0.5) / t) + self.normalize2 * K.round(t * x_neg + 0.5) / t


        x_final = x_exp_pos + x_exp_neg

        return x_final

    def compute_output_shape(self, inshape):
        return inshape[0]


class PrintTensor(Layer):
    def __init__(self, trainable=True, name=None, mess=None, **kwargs):
        super(PrintTensor, self).__init__(name=name, trainable=trainable, **kwargs)
        self.message = mess
    def build(self, inshape):
        #self.e = self.add_weight(name="e", shape=(inshape[1], inshape[2], inshape[3]), initializer=initializers.get('ones'))
        1
        # self.e = self.add_weight(name="e", shape=(inshape[-1],), initializer=initializers.get('ones'))
        # self.ee = self.add_weight(name="ee", shape=(inshape[-1],), initializer=initializers.get('ones'))
        # self.a = self.add_weight(name="a", shape=(inshape[-1],), initializer=initializers.get('ones'))
        # self.b = self.add_weight(name="b", shape=(inshape[-1],), initializer=initializers.get('zeros'))

    def call(self, x):
        x = K.print_tensor(x, message = self.message)

        return x

    def compute_output_shape(self, inshape):
        return inshape

# @keras_export('keras.layers.Dense')
# class Dense(Layer):
#     def __init__(self,
#         units,
#         activation=None,
#         use_bias=True,
#         kernel_initializer='glorot_uniform',
#         bias_initializer='zeros',
#         kernel_regularizer=None,
#         bias_regularizer=None,
#         **kwargs):
#         if 'input_shape' not in kwargs and 'input_dim' in kwargs:
#             kwargs['input_shape'] = (kwargs.pop('input_dim'),)

#         super(Dense, self).__init__(**kwargs)

#         self.units = int(units)
#         self.activation = activations.get(activation)
#         self.use_bias = use_bias
#         self.kernel_initializer = initializers.get(kernel_initializer)
#         self.bias_initializer = initializers.get(bias_initializer)
#         self.kernel_regularizer = regularizers.get(kernel_regularizer)
#         self.bias_regularizer = regularizers.get(bias_regularizer)

#         self.supports_masking = True
#         self.input_spec = InputSpec(min_ndim=2)

#         self.dtype = tf.float32

#     def build(self, input_shape):
#         dtype = dtypes.as_dtype(self.dtype or K.floatx())
#         if not (dtype.is_floating or dtype.is_complex):
#             raise TypeError('Unable to build `Dense` layer with non-floating point '
#                 'dtype %s' % (dtype,))
#         input_shape = tensor_shape.TensorShape(input_shape)
#         if tensor_shape.dimension_value(input_shape[-1]) is None:
#             raise ValueError('The last dimension of the inputs to `Dense` '
#                 'should be defined. Found `None`.')
#         last_dim = tensor_shape.dimension_value(input_shape[-1])
#         self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
#         self.kernel = self.add_weight(
#             'kernel',
#             shape=[last_dim, self.units],
#             initializer=self.kernel_initializer,
#             regularizer=self.kernel_regularizer,
#             dtype=self.dtype,
#             trainable=True)
#         if self.use_bias:
#             self.bias = self.add_weight(
#                 'bias',
#                 shape=[self.units,],
#                 initializer=self.bias_initializer,
#                 regularizer=self.bias_regularizer,
#                 dtype=self.dtype,
#                 trainable=True)
#         else:
#             self.bias = None
#         self.built = True

#     def call(self, inputs):
#         inputs = ops.convert_to_tensor(inputs)
#         rank = common_shapes.rank(inputs)
#         if rank > 2:
#             outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])

#             if not context.executing_eagerly():
#                 shape = inputs.shape.as_list()
#                 output_shape = shape[:-1] + [self.units]
#                 outputs.set_shape(output_shape)
#         else:
#             # if not self._mixed_precision_policy.should_cast_variables:
#             inputs = math_ops.cast(inputs, self.dtype)
#             outputs = gen_math_ops.mat_mul(inputs, self.kernel)

#         if self.use_bias:
#             outputs = nn.bias_add(outputs, self.bias)

#         if self.activation is not None:
#             return self.activation(outputs)

#         return outputs
    
#     def compute_output_shape(self, input_shape):
#         input_shape = tensor_shape.TensorShape(input_shape)
#         input_shape = input_shape.with_rank_at_least(2)
#         if tensor_shape.dimension_value(input_shape[-1]) is None:
#             raise ValueError(
#                 'The innermost dimension of input_shape must be defined, but saw: %s'
#                 % input_shape)
#         return input_shape[:-1].concatenate(self.units)

#     def get_config(self):
#         config = {
#             'units': self.units,
#             'activation': activations.serialize(self.activation),
#             'use_bias': self.use_bias,
#             'kernel_initializer': initializers.serialize(self.kernel_initializer),
#             'bias_initializer': initializers.serialize(self.bias_initializer),
#             'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
#             'bias_regularizer': regularizers.serialize(self.bias_regularizer),
#             'activity_regularizer':
#                 regularizers.serialize(self.activity_regularizer),
#             'kernel_constraint': constraints.serialize(self.kernel_constraint),
#             'bias_constraint': constraints.serialize(self.bias_constraint)
#         }
#         base_config = super(Dense, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))


class ExpNeuron2D(Layer):
    def __init__(self, norm=0.0, trainable=True, name=None, **kwargs):
        super(ExpNeuron2D, self).__init__(name=name, trainable=trainable, **kwargs)
        self.norm = norm

    def build(self, inshape):
        self.e = self.add_weight(name="e", shape=(inshape[-1],), initializer=initializers.Constant(1.0))
        # self.b = self.add_weight(name="b", shape=(inshape[-1],), initializer=initializers.Constant(1.0))
        # self.b = 1.0
        self.normalize = self.add_weight(name="normalize", shape=(inshape[-1],), initializer=initializers.Constant(self.norm))
        self.normalize2 = self.add_weight(name="normalize2", shape=(inshape[-1],), initializer=initializers.Constant(0.1))
        # self.r = self.add_weight(name="r", shape=(inshape[-1],), initializer=initializers.Constant(0.0))
        # 1

    def call(self, x):
        x_pos = K.cast(x > 0.0, tf.float32)
        # x_neg = K.cast(x < 0.00, tf.float32)
    
        # e_pos = K.cast(self.e > 1.0, tf.float32)
        # e_neg = K.cast(self.e < 1.0, tf.float32)
        # e_one = K.cast(self.e == 1.0, tf.float32)
        
        t = 10000.0

        exponent = self.e / (2-self.e)
        norm_exp = K.pow(10.0, self.normalize)
        bias = 0.2

        # exp_pos = x * x_pos
        # exp_neg = x * x_neg
        exp_pos = K.relu(x)
        exp_neg = -K.relu(-x)
        
        # self.e = K.maximum(K.minimum(self.e, 1.99), 0.01)
        # bias_exp = K.pow(bias, exponent)

        # exp = (K.pow(exp_pos + bias, exponent) - bias_exp) * norm_exp
        # exp_pos_round = K.round(t * exp_pos - 0.5) / t
        # exp_neg_round = K.round(t * exp_neg + 0.5) / t
        # exp_zero = K.cast(exp <= 0.0, tf.float32)

        # exp_round_down = K.round(exp)
        # incline_pos = 1.0# / (exponent * (K.pow((exp_pos_round + bias), exponent-1))) #0.1 * (K.pow(exp_pos + 1e-10 * x_neg, self.e / (2 - self.e) - 1) * norm_exp) #0.99 / (self.e / (2 - self.e) * K.pow(exp_pos + 1e-10 * x_neg, self.e / (2 - self.e) - 1) * norm_exp)
        # incline_neg = 0#0.3 # self.normalize2#
        
        # x_exp_pos = (incline_pos * (exp_pos - exp_pos_round) + (self.b * (K.pow(1.0 / self.b * (exp_pos_round + bias), exponent) - K.pow(1.0 / self.b * bias, exponent)))) * x_pos
        # x_exp_pos = (incline_pos * (exp_pos - exp_pos_round) + (K.pow(exp_pos_round + bias, exponent) - K.pow(bias, exponent))) * x_pos
        # x_exp_pos = exp_pos
        # x_exp_pos = self.normalize * (K.pow(1 / self.normalize * (exp_pos + bias), exponent) - K.pow(1 / self.normalize * bias, exponent)) * x_pos
        # x_exp_pos = self.b * (K.pow(1.0 / self.b * (exp_pos + bias), exponent) - K.pow(1.0 / self.b * bias, exponent)) * x_pos
        x_exp_pos = (K.pow(exp_pos + bias, exponent) - K.pow(bias, exponent)) * norm_exp * x_pos
        # x_exp_pos = (exp_pos - exp + norm_exp * K.round(t * K.pow(exp_pos + 1e-10, self.e / (2 - self.e) ) ) / t ) * x_pos
        # x_exp_pos_e_neg = exp * x_pos * e_neg

        # x_exp_pos = x_exp_pos_e_pos + x_exp_pos_e_neg# + x_exp_pos_e_one
        # x_exp_neg = incline_neg * (exp_neg - exp_neg_round) + self.normalize2 * exp_neg_round
        x_exp_neg = self.normalize2 * exp_neg

        x_final = x_exp_pos + x_exp_neg

        return x_final

    def compute_output_shape(self, inshape):
        return inshape

class Double(Layer):
    def __init__(self, trainable=True, name=None, **kwargs):
        super(Double, self).__init__(name=name, trainable=trainable, **kwargs)

    def build(self, inshape):
        # self.e = self.add_weight(name="e", shape=(inshape[-1],), initializer=initializers.Constant(1.0))
        self.normalize = self.add_weight(name="normalize", shape=(inshape[-1],), initializer=initializers.Constant(-1.0))
        # self.normalize2 = self.add_weight(name="normalize2", shape=(inshape[-1],), initializer=initializers.Constant(0.1))
        # self.r = self.add_weight(name="r", shape=(inshape[-1],), initializer=initializers.Constant(0.0))
        1

    def call(self, x):
        x_pos = K.relu(x)
        x_neg = -K.relu(-x)

        return K.pow(10.0, self.normalize) * x_pos + x_neg

    def compute_output_shape(self, inshape):
        return inshape


class softmax2(Layer):
    def __init__(self, trainable=True, name=None, **kwargs):
        super(softmax2, self).__init__(name=name, trainable=trainable, **kwargs)

    def build(self, inshape):
        # self.e = self.add_weight(name="e", shape=(inshape[-1],), initializer=initializers.Constant(1.0))
        # self.normalize = self.add_weight(name="normalize", shape=(inshape[-1],), initializer=initializers.Constant(-1.0))
        # self.normalize2 = self.add_weight(name="normalize2", shape=(inshape[-1],), initializer=initializers.Constant(0.1))
        # self.r = self.add_weight(name="r", shape=(inshape[-1],), initializer=initializers.Constant(0.0))
        1

    def call(self, x):
        x_pos = K.relu(x)
        x_neg = -K.relu(-x)

        x_exp = K.exp(x)
        x_sum = K.sum(K.exp(x), axis=-1, keepdims=True)
        x_final = K.pow(x_exp / x_sum, 0.1)

        return x_final

    def compute_output_shape(self, inshape):
        return inshape



class simpleplus(Layer):
    def __init__(self, trainable=True, name=None, **kwargs):
        super(simpleplus, self).__init__(name=name, trainable=trainable, **kwargs)

    def build(self, inshape):
        self.a = self.add_weight(name="a", shape=(inshape[-1],), initializer=initializers.get('zeros'))

    def call(self, x):
        x += self.a
        return x

    def compute_output_shape(self, inshape):
        return inshape

class simplemult(Layer):
    def __init__(self, trainable=True, name=None, **kwargs):
        super(simplemult, self).__init__(name=name, trainable=trainable, **kwargs)

    def build(self, inshape):
        self.a = self.add_weight(name="a", shape=(inshape[-1],), initializer=initializers.get('ones'))

    def call(self, x):
        x *= self.a
        return x

    def compute_output_shape(self, inshape):
        return inshape

class simplemultplus(Layer):
    def __init__(self, trainable=True, name=None, **kwargs):
        super(simplemultplus, self).__init__(name=name, trainable=trainable, **kwargs)

    def build(self, inshape):
        self.a = self.add_weight(name="a", shape=(inshape[-1],), initializer=initializers.get('ones'))
        self.b = self.add_weight(name="b", shape=(inshape[-1],), initializer=initializers.get('zeros'))

    def call(self, x):
        x = x * self.a + self.b
        return x

    def compute_output_shape(self, inshape):
        return inshape

def lr_schedule(epoch):
    lrate = 0.01
    if epoch > 50:
    #if epoch > 20:
        lrate = 0.005
    if epoch > 75:
    #if epoch > 40:    
        lrate = 0.003
    return lrate
 
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= float(255)
x_test /= float(255)

#z-score
# mean = np.mean(x_train,axis=(0,1,2,3))
# std = np.std(x_train,axis=(0,1,2,3))
# mean = np.mean(x_train,axis=(0,1))
# std = np.std(x_train,axis=(0,1))
# x_train = (x_train-mean)/(std+1e-7)
# x_test = (x_test-mean)/(std+1e-7)

num_classes = 100
y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)
 


img_rows = 28
img_cols = 28

# x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
# x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
# input_shape = (img_rows, img_cols, 1)


print ("shape: " + str(x_train.shape))

np.random.seed(0)

my_glorot_uniform = initializers.glorot_uniform(seed=0)


useEXP = True
lr_decay = 1.0
lr_multipliers = {'exp_neuron_1': lr_decay, 'exp_neuron_2': lr_decay, 'exp_neuron_3': lr_decay}#, 'exp_neuron_4': lr_decay, 'exp_neuron_5': lr_decay, 'exp_neuron_6': lr_decay}

decay_decay = 1.0
decay_multipliers = {'exp_neuron_1': decay_decay, 'exp_neuron_2': decay_decay, 'exp_neuron_3': decay_decay, 'exp_neuron_4': decay_decay, 'exp_neuron_5': decay_decay, 'exp_neuron_6': decay_decay}

acti = 'relu'

num_node = 200
num_layer = 7
# model.add(ExpNeuron2D()) 
# model.add(ExpNeuron4D()) 

alpha_init = initializers.Constant(0.1)

weight_decay = 1e-4

inp = Input(shape=x_train.shape[1:])
norm = 1.89
#0
flatten = Flatten()(inp)

act = flatten
# dense = Dense(3072, kernel_initializer=my_glorot_uniform)(act)
# act = PReLU(alpha_initializer=alpha_init)(dense)
# dense = myDense(10, kernel_initializer=my_glorot_uniform)(act)
# act = ExtendedRelu(norm)(dense)
for i in range(num_layer):
    # model.add(keras.layers.Lambda(lambda x: tf.Print(x, [x], message='before1(' + str(i) + '): ', first_n=-1, summarize=50)))
    if i == 0:
        dense = myDense(1000, exp=False, kernel_initializer=my_glorot_uniform)(act)
    else:
        dense = myDense(1000, exp=True, kernel_initializer=my_glorot_uniform)(act)

    # if i == num_layer - 1:
    #     dense = keras.layers.Lambda(lambda x: tf.Print(x, [x], message='dense: ', first_n=-1, summarize=110))(dense)
    # model.add(keras.layers.Lambda(lambda x: tf.Print(x, [x], message='before2(' + str(i) + '): ', first_n=-1, summarize=50)))
    # bn = myBatchNormalization(10)(dense)
    # bn = BatchNormalization()(dense)
    bn = BN()(dense)
    # model.add(PrintTensor(mess=str(i) + ': '))
    # model.add(keras.layers.Lambda(lambda x: tf.Print(x, [x], message='before3(' + str(i) + '): ', first_n=-1, summarize=50)))
    # model.add(Activation('relu'))
    # model.add(Double())
    
    # if i == num_layer - 1:
    # act = PReLU(alpha_initializer=alpha_init)(bn)
    # act = Double()(act1)
    # else:
    act = PReLU(alpha_initializer=alpha_init)(bn)
    # model.add(ExpNeuron2D()) 
    # act = Activation('elu')(dense)
    # if i == num_layer - 1:
    # act = ExpNeuron2D()(bn)
    # else:
    #     act = ExpNeuron2D(3.0)(bn)
    # act = ExpNorm()(act1)
    # pr = keras.layers.Lambda(lambda x: tf.Print(x, [x], message='before2(' + str(i) + '): ', first_n=-1, summarize=50))(act)
    # act = ExtendedRelu(norm)(dense)
    # bn = BatchNormalization()(act)

    # model.add(keras.layers.Lambda(lambda x: tf.Print(x, [x], message='after(' + str(i) + '): ', first_n=-1, summarize=50)))
# dense = Dense(3072, kernel_initializer=my_glorot_uniform)(act)
# act = PReLU(alpha_initializer=alpha_init)(dense)
last_dense = myDense(num_classes, exp=True, kernel_initializer=my_glorot_uniform)(act)
# pr = keras.layers.Lambda(lambda x: tf.Print(x, [x], message='last_dense: ', first_n=-1, summarize=110))(last_dense)
# model.add(BatchNormalization())
# model.add(ExpNeuron2D()) 
# model.add(PReLU(alpha_initializer=alpha_init))
softmax = Activation('softmax')(last_dense)
# softmax = softmax2()(last_dense)
model = Model(inputs=inp, outputs = softmax)
model.summary()



# model = Sequential()

# #0
# model.add(Conv2D(32, (3,3), padding='same', kernel_initializer=my_glorot_uniform, input_shape=x_train.shape[1:]))#, kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
# model.add(BatchNormalization())
# # model.add(Activation(acti))

# # model.add(PReLU(alpha_initializer=alpha_init))
# model.add(ExpNeuron2D()) 
# # model.add(Extended2D()) 

# #3
# model.add(Conv2D(32, (3,3), padding='same', kernel_initializer=my_glorot_uniform))#, kernel_regularizer=regularizers.l2(weight_decay)))
# model.add(BatchNormalization())
# # model.add(Activation(acti))
# # model.add(PReLU(alpha_initializer=alpha_init))
# model.add(ExpNeuron2D()) 
# # model.add(Extended2D()) 
# model.add(MaxPooling2D(pool_size=(2,2)))
# # model.add(Dropout(0.3, seed=0))

# #8
# model.add(Conv2D(64, (3,3), padding='same', kernel_initializer=my_glorot_uniform))#, kernel_regularizer=regularizers.l2(weight_decay)))
# model.add(BatchNormalization())
# # model.add(Activation(acti))
# # model.add(PReLU(alpha_initializer=alpha_init))
# model.add(ExpNeuron2D())
# # model.add(Extended2D()) 

# #11
# model.add(Conv2D(64, (3,3), padding='same', kernel_initializer=my_glorot_uniform))#, kernel_regularizer=regularizers.l2(weight_decay)))\
# model.add(BatchNormalization())
# # model.add(Activation(acti))
# # model.add(PReLU(alpha_initializer=alpha_init))
# model.add(ExpNeuron2D())
# # model.add(Extended2D())  
# model.add(MaxPooling2D(pool_size=(2,2)))
# # model.add(Dropout(0.3, seed=0))
 
# #16
# model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=my_glorot_uniform))#, kernel_regularizer=regularizers.l2(weight_decay)))
# model.add(BatchNormalization())
# # model.add(Activation(acti))
# # model.add(PReLU(alpha_initializer=alpha_init))
# model.add(ExpNeuron2D())
# # model.add(Extended2D())  

# #19
# model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=my_glorot_uniform))#, kernel_regularizer=regularizers.l2(weight_decay)))
# model.add(BatchNormalization())
# # model.add(Activation(acti))
# # model.add(PReLU(alpha_initializer=alpha_init))
# model.add(ExpNeuron2D())
# # model.add(Extended2D())  
# model.add(MaxPooling2D(pool_size=(2,2)))
# # model.add(Dropout(0.3, seed=0))
 
# #24
# model.add(Flatten())
# # model.add(Dense(100, kernel_initializer=my_glorot_uniform)) 
# # model.add(BatchNormalization())
# # model.add(Activation('relu'))


# # model.add(Dense(100, kernel_initializer=my_glorot_uniform)) 
# # model.add(BatchNormalization())
# # model.add(Activation('relu'))


# model.add(Dense(num_classes, kernel_initializer=my_glorot_uniform)) 
# # model.add(BatchNormalization())
# model.add(ExpNeuron2D()) 
# model.add(Activation('softmax'))
# model.summary()

 
#data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )
np.random.seed(0)
datagen.fit(x_train)
# f = open("weights.txt", 'w')
# print_weights = LambdaCallback(on_batch_begin=lambda batch,logs: print(str(model.layers[8].get_weights())))
print_weights2 = LambdaCallback(on_batch_begin=lambda batch,logs: print(str(model.layers[5].get_weights())))
print_weights3 = LambdaCallback(on_batch_begin=lambda batch,logs: print(str(model.layers[2 + 3 * 30].get_weights())))
print_weights4 = LambdaCallback(on_batch_begin=lambda batch,logs: print(str(model.layers[2 + 3 * 199].get_weights())))
 
#opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
opt_rms = LearningRateMultiplier(SGD, lr_multipliers=lr_multipliers, decay_multipliers=decay_multipliers, lr=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
np.random.seed(0)
hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size, epochs=100,
                    verbose=1,validation_data=(x_test,y_test),callbacks=[LearningRateScheduler(lr_schedule)],#, print_weights2, print_weights3, print_weights4],
                    shuffle=False)
tr_loss = hist.history['loss']
tr_acc = hist.history['acc']

val_loss = hist.history['val_loss']
min_loss = min(val_loss)
min_index = val_loss.index(min_loss)
val_acc = hist.history['val_acc']
min_acc = val_acc[min_index]

print (tr_loss)
print (tr_acc)
print (val_loss)
print (val_acc)

print ("Epoch: " + str(min_index + 1))
print ("Min loss: " + str(min_loss))
print ("Opt acc: " + str(min_acc))

# # 259 12 16 19 
# print (model.layers[2].get_weights())
# print (model.layers[5].get_weights())
# print (model.layers[2 + 3*30].get_weights())
# print (model.layers[6].get_weights())
# print (model.layers[8].get_weights())
# print (model.layers[10].get_weights())
# print (model.layers[3].get_weights())
# print (model.layers[5].get_weights())
# print (model.layers[7].get_weights())
# print (model.layers[9].get_weights())
# print (model.layers[11].get_weights())
# print (model.layers[13].get_weights())
# print (model.layers[15].get_weights())
# print (model.layers[17].get_weights())
# print (model.layers[19].get_weights())
# print (model.layers[21].get_weights())
# print (model.layers[23].get_weights())
# print (model.layers[25].get_weights())
# print (model.layers[27].get_weights())
# print (model.layers[29].get_weights())
# print (model.layers[31].get_weights())
# print (model.layers[33].get_weights())
# print (model.layers[35].get_weights())
# print (model.layers[37].get_weights())
# print (model.layers[39].get_weights())
# print (model.layers[41].get_weights())
# print (model.layers[43].get_weights())
# print (model.layers[45].get_weights())
# print (model.layers[47].get_weights())
# print (model.layers[49].get_weights())
# print (model.layers[51].get_weights())
# print (model.layers[53].get_weights())
# print (model.layers[55].get_weights())
# print (model.layers[57].get_weights())
# print (model.layers[59].get_weights())
# print (model.layers[61].get_weights())
# print (model.layers[63].get_weights())
# print (model.layers[65].get_weights())
# print (model.layers[67].get_weights())
# print (model.layers[69].get_weights())
# print (model.layers[71].get_weights())
# print (model.layers[73].get_weights())
# print (model.layers[75].get_weights())
# print (model.layers[77].get_weights())
# print (model.layers[79].get_weights())
# print (model.layers[81].get_weights())
# print (model.layers[83].get_weights())
# print (model.layers[85].get_weights())
# print (model.layers[87].get_weights())
# print (model.layers[89].get_weights())
# print (model.layers[91].get_weights())
# print (model.layers[93].get_weights())
# print (model.layers[95].get_weights())
# print (model.layers[97].get_weights())
# print (model.layers[99].get_weights())
# print (model.layers[101].get_weights())
# print (model.layers[103].get_weights())
# print (model.layers[105].get_weights())
# print (model.layers[107].get_weights())
# print (model.layers[109].get_weights())
# print (model.layers[111].get_weights())
# print (model.layers[113].get_weights())
# print (model.layers[115].get_weights())
# print (model.layers[117].get_weights())
# print (model.layers[119].get_weights())
# print (model.layers[121].get_weights())
# print (model.layers[123].get_weights())
# print (model.layers[125].get_weights())
# print (model.layers[127].get_weights())
# print (model.layers[129].get_weights())
# print (model.layers[131].get_weights())
# print (model.layers[133].get_weights())
# print (model.layers[135].get_weights())
# print (model.layers[137].get_weights())
# print (model.layers[139].get_weights())
# print (model.layers[141].get_weights())
# print model.layers[3].get_weights()
# print model.layers[6].get_weights()
# print model.layers[9].get_weights()
# print model.layers[12].get_weights()
# print model.layers[15].get_weights()
# print model.layers[204].get_weights()
# print model.layers[207].get_weights()
# print model.layers[210].get_weights()


# print model.layers[9].get_weights()

# print model.layers[18].get_weights()
# print model.layers[20].get_weights()
# print model.layers[27].get_weights()

#save to disk
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model.h5') 
 
#testing
scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0]))


##############################################################################

# import torch
# import torchvision
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.nn.modules import dropout, batchnorm


# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=100)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=100)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(32, (3, 3))
#         self.conv2 = nn.Conv2d(32, (3, 3))
#         self.pool = nn.MaxPool2d(2, 2)


#         # self.conv1 = nn.Conv2d(3, 6, 5)
#         # self.pool = nn.MaxPool2d(2, 2)
#         # self.conv2 = nn.Conv2d(6, 16, 5)
#         # self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         # self.fc2 = nn.Linear(120, 84)
#         # self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()


# learning_rate = 0.001
# l2 = 0

# print learning_rate
# print l2

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

# net = Net()
# net = net.to(device)

# criterion = nn.CrossEntropyLoss()
# #optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=l2, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=l2)

# for epoch in range(10):  # loop over the dataset multiple times
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         #inputs, labels = data
#         inputs, labels = data[0].to(device), data[1].to(device)

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0

#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data[0].to(device), data[1].to(device)
#             outputs = net(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     print('Accuracy of the network on the 10000 test images: %d %%' % (
#         100 * correct / total))

# print('Finished Training')

# dataiter = iter(testloader)
# nextData = dataiter.next()
# images, labels = nextData[0].to(device), nextData[1].to(device)

# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# outputs = net(images)

# _, predicted = torch.max(outputs, 1)

# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                               for j in range(4)))

# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data[0].to(device), data[1].to(device)
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Accuracy of the network on the 10000 test images: %d %%' % (
#     100 * correct / total))

# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data[0].to(device), data[1].to(device)
#         outputs = net(images)
#         _, predicted = torch.max(outputs, 1)
#         c = (predicted == labels).squeeze()
#         for i in range(4):
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1


# for i in range(10):
#     print('Accuracy of %5s : %2d %%' % (
#         classes[i], 100 * class_correct[i] / class_total[i]))

# del dataiter