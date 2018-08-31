import numpy as np
from data2array import data2array
from image_embedding import ime_model
import copy

img_size = (64, 64, 3)
weights = 'DenseNet.h5'
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
            y1.append(train_list[i]['label_attribute'])
            _y2 = np.zeros(230)
            _y2[train_list[i]['label_array']] = 1
            y2.append(_y2)
            y3.append(train_list[i]['label_real_name_class_wordembeddings'])

        x = np.array(x)
        y1 = np.array(y1)
        y2 = np.array(y2)
        y3 = np.array(y3)
        y4 = copy.deepcopy(y2)

        model = self.model(lr=lr)
        if load_w:
            model.load_weights(self.model_weights)
        model.fit(x=x[:train_num], y=[y1[:train_num], y2[:train_num], y3[:train_num], y4[:train_num]],
                  validation_data=[x[train_num:-val_num],
                                   [y1[train_num:-val_num], y2[train_num:-val_num], y3[train_num:-val_num],
                                    y4[train_num:-val_num]]],
                  epochs=epochs, batch_size=batch_size)
        model_weights = copy.deepcopy(self.model_weights)
        model.save(self.model_weights)
        ev = model.evaluate(x=x[-val_num:], y=[y1[-val_num:], y2[-val_num:], y3[-val_num:], y4[-val_num:]],
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
        _, __, ___, predict = model.predict(np.array(test_list_array))
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


ime = Ime(base_path=path, model_weights=weights)

ime.train(lr=0.1, epochs=3, batch_size=123, load_w=0)

# ime.submit()
