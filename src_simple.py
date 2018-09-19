from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense, Concatenate, ZeroPadding2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Model, Sequential
from keras import backend as K
from keras.applications import VGG16, VGG19, ResNet50, DenseNet201, DenseNet121, InceptionV3, Xception, \
    InceptionResNetV2
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras import layers
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.optimizers import Adam, RMSprop
from keras import losses
from keras import metrics
import numpy as np
import pickle
from data2array import data2array
import copy

np.random.seed(123)
img_size = (64, 64, 3)
# weights = 'DenseNet121_Xception_x_32.h5'
# weights = 'DenseNet201_x_32.h5'
# weights = 'DenseNet121_x_32.h5'
weights = 'DenseNet121.h5'

path = 'C:/Users/99263/Downloads/lyb/'


# path = '/Users/mahaoyang/Downloads/'


def model_cnn(lr):
    inputs = Input(shape=(img_size[0], img_size[1], img_size[2]))
    base_model = DenseNet121(input_tensor=inputs, weights=None, include_top=False)
    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    x = Dense(30, activation='relu')(x)
    predictions = Dense(230, activation='softmax')(x)

    opti = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=opti, loss=losses.categorical_crossentropy, metrics=[metrics.categorical_accuracy])
    model.summary()
    return model


class SimpleNN(object):
    def __init__(self, base_path, model_weights):
        self.base_path = base_path
        self.model_weights = model_weights

    @staticmethod
    def model(lr):
        return model_cnn(lr)

    def train(self, lr, epochs, batch_size):
        model = self.model(lr)
        data = data2array(self.base_path)
        train_list = data['train_list']
        train_num = 30000
        x = []
        y = []
        for i in train_list:
            x.append(train_list[i]['img_array'])
            temp = np.zeros((230,))
            temp[train_list[i]['label_array']] = 1
            y.append(temp)
        x = np.array(x)
        y = np.array(y)

        model.fit(x=x[:train_num], y=y[:train_num], validation_split=0.2, epochs=epochs, batch_size=batch_size)
        model.save(self.model_weights)

        eva = model.evaluate(x=x[train_num:], y=y[train_num:])
        print(eva)
        return model

    def submit(self, lr):
        model = self.model(lr)
        data = data2array(self.base_path)
        test_list = data['test_list']
        model.load_weights(self.model_weights)
        submit_lines = []
        for i in test_list:
            test_list[i]['label_array'] = model.predict(np.array([test_list[i]['img_array']]))
            max_index = int(np.where(test_list[i]['label_array'] == np.max(test_list[i]['label_array']))[1][0])
            test_list[i]['label'] = data['label_map'][max_index]
            submit_lines.append([i, test_list[i]['label']])

        for i in submit_lines:
            with open('submit.txt', 'a') as f:
                f.write('%s\t%s\n' % (i[0], i[1]))




def model_3(lr=0.00001):
    inputs = Input(shape=(img_size[0], img_size[1], img_size[2]))

    inputs_p = ZeroPadding2D(padding=((21, 21), (21, 21)))(inputs)
    base_model1 = DenseNet121(input_tensor=inputs_p, weights=None, include_top=False)
    base_model2 = VGG19(input_tensor=inputs_p, include_top=False, weights=None)
    base_model3 = InceptionResNetV2(input_tensor=inputs_p, include_top=False, weights=None)
    x1 = GlobalMaxPooling2D()(base_model1.output)
    x2 = GlobalMaxPooling2D()(base_model2.output)
    x2 = BatchNormalization(epsilon=1e-6, weights=None)(x2)
    x3 = GlobalMaxPooling2D()(base_model3.output)
    x3 = BatchNormalization(epsilon=1e-6, weights=None)(x3)
    x_1 = Dense(300, activation='elu')(x1)
    x_2 = Dense(30, activation='sigmoid')(x2)

    # base_model = DenseNet121(input_tensor=inputs, weights=None, include_top=False)
    # xc = Conv2D(4096, 2, activation='relu', padding='same')(base_model.output)
    # x1 = GlobalMaxPooling2D()(xc)
    # x1 = BatchNormalization(epsilon=1e-6, weights=None)(x1)
    # x_1 = Dense(300, activation='elu')(x1)
    # x2 = GlobalMaxPooling2D()(xc)
    # x2 = BatchNormalization(epsilon=1e-6, weights=None)(x2)
    # x_2 = Dense(30, activation='sigmoid')(x2)
    # x3 = GlobalMaxPooling2D()(xc)
    # x3 = BatchNormalization(epsilon=1e-6, weights=None)(x3)

    mg = Concatenate(axis=1)([x1, x2, x3])
    x3 = Dropout(0.5)(mg)
    x_3 = Dense(230, activation='softmax')(x3)
    # x = Concatenate(axis=1)([x, x2])
    # predictions = Dense(300)(x)

    model = Model(inputs=inputs, outputs=[x_1, x_2, x_3])
    opti = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # opti = RMSprop(lr=lr, rho=0.9, epsilon=1e-13)
    model.compile(optimizer=opti,
                  loss=[losses.mean_squared_error, losses.categorical_crossentropy, losses.categorical_crossentropy],
                  metrics=[metrics.categorical_accuracy])
    model.summary()
    return model


def distance(vec1, vec2):
    sub = np.square(np.array(vec1) - np.array(vec2).astype('float32'))
    # print(vec1)
    return np.sqrt(np.sum(sub))


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


class PWNN(SimpleNN):
    @staticmethod
    def model(lr=0.000001, epochs=10, batch_size=23):
        return model_3(lr=lr)

    def train(self, lr=0.000001, epochs=10, batch_size=23, rstep=10, start_rstep=-1):
        start_rstep += 1
        model_weights = copy.deepcopy(self.model_weights)
        data = data2array(self.base_path)
        train_list = data['train_list']
        train_num = 30000
        x = []
        y1, y2, y3 = [], [], []
        for i in train_list:
            x.append(train_list[i]['img_array'])
            y1.append(train_list[i]['label_real_name_class_wordembeddings'])
            y2.append(train_list[i]['label_attribute'])
            _y3 = np.zeros(230)
            _y3[train_list[i]['label_array']] = 1
            y3.append(_y3)
        x = np.array(x)
        y1 = np.array(y1)
        y2 = np.array(y2)
        y3 = np.array(y3)

        model = self.model(lr=lr)
        for i in range(start_rstep, start_rstep + rstep):
            print('\nround : %s\n' % i)
            try:
                model.load_weights('%s_rstep_%s_%s' % (self.model_weights, rstep, start_rstep - 1))
            except Exception:
                print('\nNot have weights yet!\n')
            model.fit(x=x[:train_num], y=[y1[:train_num], y2[:train_num], y3[:train_num], ],
                      validation_data=[x[train_num:-200], [y1[train_num:-200], y2[train_num:-200], y3[train_num:-200]]],
                      epochs=epochs, batch_size=batch_size)
            model_weights = copy.deepcopy('%s_rstep_%s_end_%s.h5' % (self.model_weights, rstep, i))
            model.save('%s_rstep_%s_end_%s.h5' % (self.model_weights, rstep, i))
        ev = model.evaluate(x=x[-200:], y=[y1[-200:], y2[-200:], y3[-200:]], batch_size=200)
        ev = dict(zip(model.metrics_names, ev))
        print(ev)
        return model_weights

    def submit(self, model_weights):
        data = data2array(self.base_path)
        test_list_array = data['test_list_array']
        test_list_name = data['test_list_name']
        model = self.model()
        model.load_weights(model_weights)
        _, __, predict = model.predict(np.array(test_list_array))
        submit_lines = []
        n = 0
        for i in predict:
            m = np.where(i == np.max(i))
            max_index = int(m[0])
            lable = data['label_map'][max_index]
            submit_lines.append([test_list_name[n], lable])
            n = n + 1
        submit = ''
        for i in submit_lines:
            submit += '%s\t%s\n' % (i[0], i[1])
        with open('submit.txt', 'w') as f:
            f.write(submit)

    def submit_pw(self):
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


def attention_2d_block(inputs):
    dim = int(inputs.shape[1])
    a = layers.Dense(dim, activation='softmax')(inputs)
    output_attention_mul = layers.multiply([inputs, a])
    return output_attention_mul


def res_dense_block(inputs, dim):
    a = attention_2d_block(inputs)
    a = layers.BatchNormalization()(a)
    a = layers.GaussianNoise(0.01)(a)
    a = layers.Concatenate()([inputs, a])
    a = layers.Dense(dim)(a)
    a = layers.BatchNormalization()(a)
    return a

class MixNN(SimpleNN):
    @staticmethod
    def model(lr):
        return model_mix(lr)

    def train(self, lr, epochs, batch_size):
        model = self.model(lr)
        data = data2array(self.base_path)
        train_list = data['train_list']
        train_num = 36000
        x = []
        wx = []
        y = []
        for i in train_list:
            x.append(train_list[i]['img_array'])
            wx.append(train_list[i]['label_real_name_class_wordembeddings'])
            temp = np.zeros((230,))
            temp[train_list[i]['label_array']] = 1
            y.append(temp)
        x = np.array(x)
        wx = np.array(wx).astype('float64')
        y = np.array(y)

        # x = np.random.shuffle(x)
        # wx = np.random.shuffle(wx)
        # y = np.random.shuffle(y)

        model.fit(x=x[:train_num], y=[y[:train_num], wx[:train_num]], validation_split=0.2, epochs=epochs,
                  batch_size=batch_size)
        model.save(self.model_weights)

        eva = model.evaluate(x=x[train_num:], y=[y[train_num:], wx[train_num:]])
        print(eva)
        return model


def model_mix(lr):
    inputs = Input(shape=(img_size[0], img_size[1], img_size[2]))
    base_model = DenseNet121(input_tensor=inputs, weights=None, include_top=False)
    x = base_model.output
    # x = layers.GaussianDropout(0.01)(x)
    img_features = Flatten()(x)
    img_features = layers.BatchNormalization()(img_features)

    w0 = res_dense_block(img_features, 8)
    w1 = res_dense_block(w0, 16)
    w2 = res_dense_block(w1, 32)
    w_out = res_dense_block(w2, 50)
    w3 = res_dense_block(w_out, 64)
    w4 = res_dense_block(w3, 128)

    p0 = attention_2d_block(w4)
    p0 = layers.BatchNormalization()(p0)
    p0 = layers.GaussianNoise(0.01)(p0)
    p0 = layers.Concatenate()([p0, w4])

    predictions = Dense(230, activation='softmax')(p0)

    opti = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model = Model(inputs=base_model.input, outputs=[predictions, w_out])
    model.compile(optimizer=opti, loss=[losses.categorical_crossentropy, losses.mean_squared_logarithmic_error],
                  metrics=['accuracy'])
    model.summary()
    return model


if __name__ == '__main__':
    # nn = SimpleNN(base_path=path, model_weights=weights)
    # nn.train(1e-1, 5, 100)
    # nn.submit(1e-1)
    nn = MixNN(base_path=path, model_weights=weights)
    nn.train(1e-0, 30, 100)
    # model_3()
    # nn = PWNN(base_path=path, model_weights=weights)
    # model_weights = nn.train(lr=0.000001, epochs=3, batch_size=123, rstep=7, start_rstep=-1)
    # nn.submit(model_weights=model_weights)
    # nn.submit(model_weights=model_weights)
