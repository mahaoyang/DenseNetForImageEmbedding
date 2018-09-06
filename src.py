import numpy as np
from data2array import data2array
from image_embedding import ime_model, raw_model
import xgboost as xgb
import copy
import keras

img_size = (64, 64, 3)
weights = 'DenseNet_2.h5'
path = 'D:/lyb/'


class Ime:
    def __init__(self, base_path, model_weights):
        self.base_path = base_path
        self.model_weights = model_weights

    @staticmethod
    def model(lr=0.000001):
        return ime_model(lr=lr, shape=img_size)

    def train(self, lr=0.000001, epochs=10, batch_size=23, load_w=0):
        data = data2array(self.base_path)
        train_list = data['train_list']
        train_num = 30000
        val_num = 1000
        x = []
        y1, y2, y3 = [], [], []
        for i in train_list:
            x.append(train_list[i]['img_array'])
            y1.append(train_list[i]['attributes_per_classf_330_num'])
            # _y1 = np.zeros(230)
            # _y1[train_list[i]['label_array']] = 1
            _y2 = data['label_map'].index(train_list[i]['label'])
            y2.append(_y2)
            # y3.append(train_list[i]['label_real_name_class_wordembeddings'])

        x = np.array(x)
        y1 = np.array(y1)
        y2 = np.array(y2)
        # y3 = np.array(y3)
        # y4 = copy.deepcopy(y2)

        # y1 = keras.utils.np_utils.to_categorical(y1, 219)
        # y2 = keras.utils.np_utils.to_categorical(y2, 230)
        # y4 = keras.utils.np_utils.to_categorical(y4, 230)

        data_gen = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=True, featurewise_std_normalization=True, rotation_range=20, width_shift_range=0.2,
            height_shift_range=0.2, horizontal_flip=True)
        data_gen.fit(x[:train_num], augment=True, rounds=1)

        validation_generator = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=True, featurewise_std_normalization=True, rotation_range=20, width_shift_range=0.2,
            height_shift_range=0.2, horizontal_flip=True)
        validation_generator.fit(x[train_num:-val_num])

        model = self.model(lr=lr)
        if load_w:
            model.load_weights(self.model_weights)

        # reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
        # tb_cb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=False,
        #                                     embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        # es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.09, patience=5, verbose=0, mode='auto')
        # cbks = []
        # cbks.append(tb_cb)
        # cbks.append(es_cb)
        # cbks.append(reduce_lr_loss)

        model.fit_generator(
            data_gen.flow(x=x[:train_num], y=[y1[:train_num], y2[:train_num]], batch_size=batch_size),
            steps_per_epoch=len(x), epochs=epochs, validation_data=validation_generator.flow(
                x=x[train_num:-val_num], y=[y1[train_num:-val_num], y2[train_num:-val_num]],
                batch_size=batch_size), validation_steps=800)  # , callbacks=cbks
        model_weights = copy.deepcopy(self.model_weights)
        model.save(self.model_weights)
        ev = model.evaluate(x=x[-val_num:], y=[y1[-val_num:], y2[-val_num:]],
                            batch_size=200)
        ev = dict(zip(model.metrics_names, ev))
        print(ev)
        return model_weights

    def submit(self):
        data = data2array(self.base_path)
        test_list_array = data['test_list_array']
        test_list_name = data['test_list_name']
        model = self.model()
        model.load_weights(self.model_weights)
        _, predict = model.predict(np.array(test_list_array))
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

    @staticmethod
    def main():
        ime = Ime(base_path=path, model_weights=weights)

        ime.train(lr=10, epochs=1, batch_size=233, load_w=0)

        # ime.submit()


class RawIme:
    def __init__(self, base_path, model_weights):
        self.base_path = base_path
        self.model_weights = model_weights

    @staticmethod
    def model(lr=0.000001):
        return raw_model(lr=lr, shape=img_size)

    def train(self, lr=0.000001, epochs=10, batch_size=23, load_w=0):
        data = data2array(self.base_path)
        train_list = data['train_list']
        train_num = 30000
        val_num = 1000
        x = []
        y1, y2, y3 = [], [], []
        for i in train_list:
            x.append(train_list[i]['img_array'])
            y2.append(train_list[i]['attributes_per_classf_330_num'])
            # _y1 = np.zeros(230)
            # _y1[train_list[i]['label_array']] = 1
            _y1 = data['label_map'].index(train_list[i]['label'])
            y1.append(_y1)
            # y3.append(train_list[i]['label_real_name_class_wordembeddings'])

        x = np.array(x)
        y1 = np.array(y1)
        # y2 = np.array(y2)
        # y3 = np.array(y3)
        # y4 = copy.deepcopy(y2)

        y1 = keras.utils.np_utils.to_categorical(y1, 230)
        # y2 = keras.utils.np_utils.to_categorical(y2, 230)
        # y4 = keras.utils.np_utils.to_categorical(y4, 230)

        data_gen = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=True, featurewise_std_normalization=True, rotation_range=20, width_shift_range=0.2,
            height_shift_range=0.2, horizontal_flip=True)
        data_gen.fit(x[:train_num], augment=True, rounds=1)

        validation_generator = keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=True, featurewise_std_normalization=True, rotation_range=20, width_shift_range=0.2,
            height_shift_range=0.2, horizontal_flip=True)
        validation_generator.fit(x[train_num:-val_num])

        model = self.model(lr=lr)[0]
        if load_w:
            model.load_weights(self.model_weights)

        # reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
        # tb_cb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=False,
        #                                     embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        # es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.09, patience=5, verbose=0, mode='auto')
        # cbks = []
        # cbks.append(tb_cb)
        # cbks.append(es_cb)
        # cbks.append(reduce_lr_loss)

        model.fit_generator(
            data_gen.flow(x=x[:train_num], y=y1[:train_num], batch_size=batch_size),
            steps_per_epoch=len(x), epochs=epochs, validation_data=validation_generator.flow(
                x=x[train_num:-val_num], y=y1[train_num:-val_num], batch_size=batch_size),
            validation_steps=800)  # , callbacks=cbks
        model_weights = copy.deepcopy(self.model_weights)
        model.save(self.model_weights)
        ev = model.evaluate(x=x[-val_num:], y=y1[-val_num:], batch_size=200)
        ev = dict(zip(model.metrics_names, ev))
        print(ev)
        return model_weights

    def train_xgb(self):
        data = data2array(self.base_path)
        train_list = data['train_list']
        train_num = 30000
        val_num = 1000
        x = []
        y1, y2, y3 = [], [], []
        for i in train_list:
            x.append(train_list[i]['img_array'])
            y2.append(train_list[i]['attributes_per_classf_330_num'])
            # _y1 = np.zeros(230)
            # _y1[train_list[i]['label_array']] = 1
            _y1 = data['label_map'].index(train_list[i]['label'])
            y1.append(_y1)
            # y3.append(train_list[i]['label_real_name_class_wordembeddings'])
        x = np.array(x)
        y1 = np.array(y1)
        model = self.model()
        model_230 = model[0]
        model_middle_layer = model[1]
        model_230.load_weights(self.model_weights)
        predict = model_middle_layer.predict(np.array(x))
        params = {
            'booster': 'gbtree',
            # 这里手写数字是0-9，是一个多类的问题，因此采用了multisoft多分类器，
            'objective': 'multi:softmax',
            'num_class': 230,  # 类数，与 multisoftmax 并用
            'gamma': 0.05,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
            'max_depth': 230,  # 构建树的深度 [1:]
            # 'lambda':450,  # L2 正则项权重
            'subsample': 0.4,  # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
            'colsample_bytree': 0.7,  # 构建树树时的采样比率 (0:1]
            # 'min_child_weight':12, # 节点的最少特征数
            'silent': 1,
            'eta': 0.05,  # 如同学习率
            'seed': 710,
            'nthread': 15,  # cpu 线程数,根据自己U的个数适当调整
        }
        plst = params
        # Using 10000 rows for early stopping.
        offset = 50000  # 训练集中数据60000，划分50000用作训练，10000用作验证

        num_rounds = 200  # 迭代你次数


        # 划分训练集与验证集
        xgtrain = xgb.DMatrix(predict[:train_num], y1[:train_num])
        xgval = xgb.DMatrix(predict[train_num:], label=y1[train_num:])

        # return 训练和验证的错误率
        watchlist = [(xgtrain, 'train'), (xgval, 'val')]

        # training model
        # early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
        model_xgb = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
        # model.save_model('./model/xgb.model') # 用于存储训练出的模型
        model_xgb.save_model('xgb.model')


    def submit(self):
        data = data2array(self.base_path)
        test_list_array = data['test_list_array']
        test_list_name = data['test_list_name']
        model = self.model()
        model_230 = model[0]
        model = model[1]
        model_230.load_weights(self.model_weights)
        predict = model[1].predict(np.array(test_list_array))
        xgtest = xgb.DMatrix(predict)
        preds = model.predict(xgtest, ntree_limit=model.best_iteration)
        submit_lines = []
        lable_index = predict.argmax(1)
        n = 0
        for i in lable_index:
            lable = data['label_map'][i]
            submit_lines.append([test_list_name[n], lable])
            n = n + 1
        submit = ''
        for i in submit_lines:
            submit += '%s\t%s\n' % (i[0], i[1])
        with open('submit.txt', 'w') as f:
            f.write(submit)

    @staticmethod
    def main():
        rawime = RawIme(base_path=path, model_weights=weights)

        rawime.train(lr=0.0001, epochs=1, batch_size=233, load_w=1)

        # rawime.submit()


if __name__ == '__main__':
    rawime = RawIme(base_path=path, model_weights=weights)
    # rawime.main()
    rawime.train_xgb()
