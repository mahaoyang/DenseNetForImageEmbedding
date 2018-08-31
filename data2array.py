from keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import pickle
import os

base_path = 'D:/lyb/'


# base_path = '/Users/mahaoyang/Downloads/'


def data2array_b(path):
    with open(path + 'DatasetA_train_20180813/label_list.txt', 'r') as f:
        label_list = dict()
        label_map = []
        for line in f:
            line = line.strip('\n').split('\t')
            label_list[line[0]] = line[1]
            label_map.append(line[0])
    print('label_list', len(label_list))

    with open(path + 'DatasetA_train_20180813/class_wordembeddings.txt', 'r') as f:
        class_wordembeddings = dict()
        for i in f.readlines():
            ii = i.strip('\n').split(' ')
            class_wordembeddings[ii[0]] = ii[1:]
    print('class_wordembeddings', len(class_wordembeddings))

    with open(path + 'DatasetA_train_20180813/attributes_per_class.txt', 'r') as f:
        attributes_per_class = dict()
        for i in f.readlines():
            ii = i.strip('\n').split('\t')
            attributes_per_class[ii[0]] = ii[1:]
    print('attributes_per_class', len(attributes_per_class))

    with open(path + 'DatasetA_train_20180813/attribute_list.txt', 'r') as f:
        attribute_list = dict()
        for i in f.readlines():
            ii = i.strip('\n').split('\t')
            attribute_list[int(ii[0])] = ii[1:]
    print('attribute_list', len(attribute_list))

    if not os.path.exists('train_list.pickle'):

        with open(path + 'DatasetA_train_20180813/train.txt', 'r') as f:
            train_list = dict()
            for line in f:
                line = line.strip('\n').split('\t')
                train_list[line[0]] = dict()
                train_list[line[0]]['label'] = line[1]
                train_list[line[0]]['label_array'] = label_map.index(line[1])

        for img in train_list:
            pic = load_img(path + 'DatasetA_train_20180813/train/' + img, target_size=(64, 64))
            pic = img_to_array(pic)
            pic = pic.reshape((pic.shape[0], pic.shape[1], pic.shape[2]))
            train_list[img]['img_array'] = pic

        for i in train_list:
            label = train_list[i]['label']
            train_list[i]['label_real_name'] = label_list[label]
            train_list[i]['label_real_name_class_wordembeddings'] = class_wordembeddings[label_list[label]]
            train_list[i]['label_attribute'] = attributes_per_class[label]

        with open('train_list.pickle', 'wb') as f:
            pickle.dump(train_list, f)

    else:

        with open('train_list.pickle', 'rb') as f:
            train_list = pickle.load(f)
    print('train_list', len(train_list))

    if not os.path.exists('test_list.pickle'):
        with open(path + 'DatasetA_test_20180813/DatasetA_test/image.txt', 'r') as f:
            test_list = dict()
            for line in f:
                line = line.strip('\n').split('\t')
                test_list[line[0]] = dict()
        for img in test_list:
            pic = load_img(path + 'DatasetA_test_20180813/DatasetA_test/test/' + img, target_size=(64, 64))
            pic = img_to_array(pic)
            pic = pic.reshape((pic.shape[0], pic.shape[1], pic.shape[2]))
            test_list[img]['img_array'] = pic
        with open('test_list.pickle', 'wb') as f:
            pickle.dump(test_list, f)
    else:

        with open('test_list.pickle', 'rb') as f:
            test_list = pickle.load(f)
    print('test_list', len(test_list))

    test_list_name, test_list_array = [], []
    for i in test_list:
        test_list_name.append(i)
        test_list_array.append(test_list[i]['img_array'])

    data = {'label_list': label_list, 'label_map': label_map, 'train_list': train_list,
            'attributes_per_class': attributes_per_class, 'attribute_list': attribute_list,
            'class_wordembeddings': class_wordembeddings, 'test_list': test_list, }
    reverse_label_list = {v: k for k, v in data['label_list'].items()}
    data['reverse_label_list'] = reverse_label_list
    data['test_list_name'] = test_list_name
    data['test_list_array'] = test_list_array
    # data = pd.DataFrame(data)
    return data


def data2array(path):
    if os.path.exists('data.pickle'):
        with open('data.pickle', 'rb') as f:
            data = pickle.load(f)
    else:
        data = data2array_b(path=path)
        with open('data.pickle', 'wb') as f:
            pickle.dump(data, f)
    return data


if __name__ == '__main__':
    data2array(base_path)
