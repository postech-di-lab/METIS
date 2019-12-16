import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input, Dense, Flatten, Dropout
from keras.layers.merge import Dot, multiply, concatenate
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams
from keras import optimizers
from keras import activations
from keras import initializers
from keras import backend as K
from collections import defaultdict
from keras.engine.topology import Layer

class modifiedDense(Layer):
    def __init__(self, units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name=None, trainable=True, **kwargs):
        super(modifiedDense, self).__init__(name=name, trainable=trainable, **kwargs)
        self.units = int(units)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        last_dim = input_shape[-1]
        self.kernel = self.add_weight(
            'kernel',
            shape=[self.units, last_dim],
            initializer=self.kernel_initializer,
            trainable=True)

        self.e = self.add_weight(
            'e',
            shape=[self.units, last_dim],
            initializer=initializers.get('ones'),
            trainable=True)


        
        self.bias = self.add_weight(
            'bias',
            shape=[self.units,],
            initializer=self.bias_initializer,
            trainable=True)
        # self.built = True

    def call(self, inputs):
        #inputs = ops.convert_to_tensor(inputs)
        #outputs = gen_math_ops.mat_mul(inputs, self.kernel)
        #inputs: None * 2048
        inputs += 1e-7
        

        expanded_inputs = K.repeat(inputs, self.units) # None * 10 * 2048
        inputs_sign = K.sign(expanded_inputs) # None * 10 * 2048
        inputs_abs = K.abs(expanded_inputs) # None * 10 * 2048
        inputs_max = K.max(inputs_abs, axis=0) # 10 * 2048
        inputs_norm = inputs_abs / inputs_max # None * 10 * 2048
        inputs_exp = K.pow(inputs_norm, self.e) # None * 10 * 2048
        inputs_unnorm = inputs_exp * inputs_max # None * 10 * 2048
        exp_input = inputs_unnorm * inputs_sign # None * 10 * 2048
        
        # expanded_inputs = K.repeat(inputs, self.units) # None * 10 * 2048
        # inputs_sign = K.sign(expanded_inputs) # None * 10 * 2048
        # inputs_abs = K.abs(expanded_inputs) # None * 10 * 2048
        # inputs_exp = K.pow(inputs_abs, self.e) # None * 10 * 2048
        # exp_input = inputs_exp * inputs_sign # None * 10 * 2048


        exp_mult_input = exp_input * self.kernel # None * 10 * 2048
        outputs = K.sum(exp_mult_input, axis=2) # None * 10
        final_outputs = K.bias_add(outputs, self.bias) # None * 10
        return self.activation(final_outputs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)




def get_mapping(series):
    occurances = defaultdict(int)
    for element in series:
        occurances[element] += 1
    mapping = {}
    i = 0
    for element in occurances:
        i += 1
        mapping[element] = i

    return mapping




def get_data():
    data = pd.read_csv("ratings.csv")

    mapping_work = get_mapping(data["movieId"])

    data["movieId"] = data["movieId"].map(mapping_work)

    mapping_users = get_mapping(data["movieId"])

    data["movieId"] = data["movieId"].map(mapping_users)

    percentil_80 = np.percentile(data["timestamp"], 80)

    print(percentil_80)

    print(np.mean(data["timestamp"]<percentil_80))

    print(np.mean(data["timestamp"]>percentil_80))

    cols = ["userId", "movieId", "rating"]

    train = data[data.timestamp<percentil_80][cols]

    print(train.shape)

    test = data[data.timestamp>=percentil_80][cols]

    print(test.shape)

    max_user = max(data["userId"].tolist() )
    max_work = max(data["movieId"].tolist() )


    return train, test, max_user, max_work, mapping_work




def get_model_1(max_work, max_user):
    dim_embedddings = 30
    bias = 3
    # inputs
    w_inputs = Input(shape=(1,), dtype='int32')
    w = Embedding(max_work+1, dim_embedddings, name="work")(w_inputs)

    # context
    u_inputs = Input(shape=(1,), dtype='int32')
    u = Embedding(max_user+1, dim_embedddings, name="user")(u_inputs)
    o = multiply([w, u])
    o = Dropout(0.5)(o)
    o = Flatten()(o)
    o = Dense(1)(o)

    rec_model = Model(inputs=[w_inputs, u_inputs], outputs=o)
    #rec_model.summary()
    rec_model.compile(loss='mae', optimizer='adam', metrics=["mae"])

    return rec_model


def get_model_2(max_work, max_user):
    dim_embedddings = 30
    bias = 1
    # inputs
    w_inputs = Input(shape=(1,), dtype='int32')
    w = Embedding(max_work+1, dim_embedddings, name="work")(w_inputs)
    w_bis = Embedding(max_work + 1, bias, name="workbias")(w_inputs)

    # context
    u_inputs = Input(shape=(1,), dtype='int32')
    u = Embedding(max_user+1, dim_embedddings, name="user")(u_inputs)
    u_bis = Embedding(max_user + 1, bias, name="userbias")(u_inputs)
    o = multiply([w, u])
    o = concatenate([o, u_bis, w_bis])
    o = Dropout(0.5)(o)
    o = Flatten()(o)
    o = Dense(1)(o)

    rec_model = Model(inputs=[w_inputs, u_inputs], outputs=o)
    #rec_model.summary()
    rec_model.compile(loss='mae', optimizer='adam', metrics=["mae"])

    return rec_model

def get_model_3(max_work, max_user):
    dim_embedddings = 30
    bias = 1
    # inputs
    w_inputs = Input(shape=(1,), dtype='int32')
    w = Embedding(max_work+1, dim_embedddings, name="work")(w_inputs)
    w_bis = Embedding(max_work + 1, bias, name="workbias")(w_inputs)


    num_dense = 5
    num_node = 100


    # context
    u_inputs = Input(shape=(1,), dtype='int32')
    u = Embedding(max_user+1, dim_embedddings, name="user")(u_inputs)
    u_bis = Embedding(max_user + 1, bias, name="userbias")(u_inputs)
    o = multiply([w, u])
    o = Dropout(0.5)(o)
    o = concatenate([o, u_bis, w_bis])
    o = Flatten()(o)
    o = modifiedDense(num_node, activation="relu")(o)
    #o = Dense(num_node, activation="relu")(o)
    #o = Dense(num_node, activation="relu")(o)
    # for i in range(num_dense-1):
    #     o = Dense(num_node, activation="relu")(o)
        #o = modifiedDense(num_node, activation="relu")(o)
    o = Dense(1)(o)

    rec_model = Model(inputs=[w_inputs, u_inputs], outputs=o)
    #rec_model.summary()
    adam = optimizers.Adam(lr=0.001, decay=0.0)
    rec_model.compile(loss='mae', optimizer=adam, metrics=["mse"])

    return rec_model

def get_array(series):
    return np.array([[element] for element in series])