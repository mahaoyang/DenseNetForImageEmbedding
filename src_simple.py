from keras.models import Model, Sequential
from keras import backend as K
from keras.layers import Input
from keras.preprocessing.image import load_img, img_to_array
from keras import layers
from keras import applications
from keras.optimizers import Adam, RMSprop
from keras import losses
from keras.preprocessing import image
from keras import metrics
import numpy as np
import pickle
from data2array import data2array
from augmentation import img_pca
import copy
import random
import math

# np.random.seed(123)
img_size = (64, 64, 3)
weights = 'DenseNet121_2.h5'

path = 'C:/Users/99263/Downloads/lyb/'


# path = '/Users/mahaoyang/Downloads/'


def euclidean_distances(A, B):
    BT = B.transpose()
    # vecProd = A * BT
    vecProd = np.dot(A, BT)
    # print(vecProd)
    SqA = A ** 2
    # print(SqA)
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    # print(sumSqAEx)

    SqB = B ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ed = np.sqrt(SqED)
    return ed


def attention_2d_block(inputs):
    dim = int(inputs.shape[1])
    a = layers.Dense(dim, activation='softmax')(inputs)
    output_attention_mul = layers.multiply([inputs, a])
    return output_attention_mul


def augm(array):
    flag = int(random.randint(0, 6) / 7)
    if flag == 0:
        a = array
    if flag == 1:
        a = image.random_shift(array, -0.1, 0.1)
    if flag == 2:
        a = image.random_shear(array, 80)
    if flag == 3:
        a = image.random_zoom(array, (0.7, 1))
    if flag == 4:
        a = img_pca(array)
    if flag == 5:
        a = image.random_brightness(array, (0.05, 0.8))
    if flag == 6:
        a = image.random_rotation(array, 180)
    # if flag == 7:
    #     a = image.random_channel_shift(array, 0.05)
    return a


def dgen(data, batch_size):
    while 1:
        x, y = [], []
        for i in range(batch_size):
            index = random.randint(0, len(data) - 1)
            va = data[index]
            # x.append(va[0])
            x.append(augm(va[0]))
            y.append(va[1])
            # for ii in range(2):
            #     y.append(va[1])
        x = np.array(x)
        y = np.array(y)
        yield x, y


class MixNN(object):
    def __init__(self, base_path, model_weights):
        self.base_path = base_path
        self.model_weights = model_weights

    def submit(self):
        model = self.model()
        data = data2array(self.base_path)
        reverse_label_list = data['reverse_label_list']
        test_list = data['test_list']
        test_list_array = data['test_list_array']
        test_list_name = data['test_list_name']
        model.load_weights(self.model_weights)
        submit_lines = []
        test_list_label_array = model.predict(np.array(test_list_array))
        class_wordembed_keys = list(data['class_wordembeddings'].keys())
        class_wordembed_array = np.array(list(data['class_wordembeddings'].values())).astype('float32')
        dist = euclidean_distances(test_list_label_array, class_wordembed_array)
        n = 0
        for i in dist:
            i = np.array(i)[0].tolist()
            most_like = reverse_label_list[class_wordembed_keys[i.index(min(i))]]
            submit_lines.append([test_list_name[n], most_like])
            n += 1
        submit = ''
        for i in submit_lines:
            submit += '%s\t%s\n' % (i[0], i[1])
        with open('submit.txt', 'w') as f:
            f.write(submit)

    @staticmethod
    def model(lr=0.1):
        return model_mix(lr)

    def train(self, lr, epochs, batch_size):
        model = self.model(lr)
        data = data2array(self.base_path)
        train_list = data['train_list']
        train_num = 75000
        val_num = 2000
        x = []
        wx = []
        z = []
        # y = []
        for i in train_list:
            x.append(train_list[i]['img_array'])
            wx.append(np.array(train_list[i]['label_real_name_class_wordembeddings']).astype('float64'))
            z.append([np.array(train_list[i]['img_array']),
                      np.array(train_list[i]['label_real_name_class_wordembeddings']).astype('float64')])
            # temp = np.zeros((230,))
            # temp[train_list[i]['label_array']] = 1
            # y.append(temp)
        # x = np.array(x)
        # wx = np.array(wx).astype('float64')
        # y = np.array(y)

        # x = np.random.shuffle(x)
        # wx = np.random.shuffle(wx)
        # y = np.random.shuffle(y)

        # model.fit(x=x[:train_num], y=[y[:train_num], wx[:train_num]], validation_split=0.2, epochs=epochs,
        #           batch_size=batch_size)
        # model.fit(x=x[:train_num], y=wx[:train_num], validation_split=0.2, epochs=epochs,
        #           batch_size=batch_size)

        # model.load_weights(self.model_weights)
        model.fit_generator(dgen(z[:train_num], batch_size=batch_size), steps_per_epoch=100, epochs=epochs,
                            validation_data=dgen(z[train_num:-val_num], batch_size=batch_size), validation_steps=100)
        model.save(self.model_weights)

        # eva = model.evaluate(x=x[train_num:], y=[y[train_num:], wx[train_num:]])
        eva = model.evaluate_generator(dgen(np.array(z[val_num:]), batch_size=batch_size), steps=1000)
        print(eva)
        return model


def many_res_dense_block(inputs, headp, headplog, endplog, activation='linear'):
    for i in range(headplog, endplog):
        inputs = res_dense_block(inputs, int(headp * pow(2, i)), activation=activation)
    return inputs


def res_dense_block(inputs, dim, inputs_img, activation='linear'):
    a = attention_2d_block(inputs)
    a = layers.BatchNormalization()(a)
    a = layers.Concatenate()([inputs, a])
    a = layers.Dense(dim, activation=activation)(a)
    b = layers.Dense(max(int(math.log(int(inputs.shape[1]), 2)) * 5, 1))(inputs)
    c = layers.Dense(max(int(math.log(int(inputs.shape[1]), 2)) * 5, 1))(inputs_img)
    a = layers.Concatenate()([c, b, a])
    a = layers.BatchNormalization()(a)
    a = layers.GaussianNoise(0.3)(a)
    return a


def res_dense_block_explosive(inputs, dim, activation='linear'):
    a = attention_2d_block(inputs)
    a = layers.BatchNormalization()(a)
    a = layers.GaussianNoise(0.3)(a)
    a = layers.Concatenate()([inputs, a])
    a = layers.Dense(dim, activation=activation)(a)
    a = layers.BatchNormalization()(a)
    a = layers.Concatenate()([inputs, a])
    a = layers.BatchNormalization()(a)
    return a


def model_mix(lr):
    inputs = Input(shape=(img_size[0], img_size[1], img_size[2]))
    base_model = applications.Xception(input_tensor=inputs, weights=None, include_top=False)
    x = base_model.output
    # x = layers.GaussianDropout(0.01)(x)
    # img_features = layers.Flatten()(x)
    img_features = layers.GlobalMaxPooling2D()(x)
    img_features = layers.BatchNormalization()(img_features)

    # w00 = res_dense_block(img_features, 6)
    # w01 = res_dense_block(w00, 12)
    # w02 = res_dense_block(w01, 24)
    # w_out = res_dense_block(w02, 50)
    w0 = res_dense_block(img_features, 1, img_features)
    for i in range(1, 3):
        w0 = res_dense_block(w0, int(pow(2, i)), img_features)
    # for i in range(1, 2):
    #     w0 = res_dense_block(w0, i)
    w_out = layers.Dense(50)(w0)
    w_out = layers.BatchNormalization()(w_out)
    # w2 = w_out
    # w3 = res_dense_block(w2, 64)
    # w4 = res_dense_block(w3, 128)

    # mg = layers.Concatenate()([w3, w_out])
    # mg = layers.BatchNormalization()(mg)

    # p0 = attention_2d_block(img_features)
    # p0 = layers.BatchNormalization()(p0)
    # p0 = layers.GaussianNoise(0.1)(p0)
    # p0 = layers.Concatenate()([p0, w3])

    # predictions = Dense(230, activation='softmax')(img_features)

    opti = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # model = Model(inputs=base_model.input, outputs=[predictions, w_out])
    # model.compile(optimizer=opti, loss=[losses.categorical_crossentropy, losses.mean_squared_logarithmic_error],
    #               metrics=['accuracy'])
    model = Model(inputs=base_model.input, outputs=w_out)
    model.compile(optimizer=opti, loss=losses.mean_squared_logarithmic_error, metrics=['accuracy'])
    model.summary()
    return model


if __name__ == '__main__':
    nn = MixNN(base_path=path, model_weights=weights)
    nn.train(1e-0, epochs=10, batch_size=50)
    nn.submit()
