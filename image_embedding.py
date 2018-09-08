import keras
from keras import layers
from keras import losses
from keras import metrics
from keras import optimizers
from keras.models import Model
from keras import applications

from dsnet import dense_net


def ime_model(lr=0.001, shape=(64, 64, 3)):
    inputs = layers.Input(shape=shape)
    base = dense_net(input_tensor=inputs, blocks=[8, 12, 24, 16])

    o1 = base[1]
    o1 = layers.Dropout(0.5)(o1)
    o1 = layers.Dense(219, activation='softmax')(o1)

    o2 = base[0]
    o2 = layers.Dropout(0.5)(o2)
    o2 = layers.Dense(230, activation='softmax')(o2)

    # x = base[0]
    # o3 = layers.Dense(300, activation='elu')(x)
    # mg = layers.Concatenate(axis=1)([o1, o2, o3])

    # mg = layers.Concatenate(axis=1)([o1, o2])
    # o4 = layers.Activation('relu')(mg)
    # o4 = layers.BatchNormalization(axis=1, epsilon=1.001e-5)(o4)
    # o4 = layers.Dropout(0.5)(o4)
    # o4 = layers.Dense(230, activation='softmax')(o4)

    model = Model(inputs=inputs, outputs=[o1, o2])

    opti = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(optimizer=opti,
                  loss=[losses.categorical_crossentropy, losses.categorical_crossentropy],
                  metrics=[metrics.categorical_accuracy])
    model.summary()
    return model


def raw_model(lr=0.001, shape=(64, 64, 3)):
    inputs = layers.Input(shape=shape)
    base = applications.DenseNet121(input_tensor=inputs, weights='imagenet', include_top=False)
    for i, layer in enumerate(base.layers):
        print(i, layer.name)

    # for layer in base.layers[:-7]:
    #    layer.trainable = False
    # for layer in base.layers[-7:]:
    #    layer.trainable = True
    output = layers.GlobalMaxPooling2D()(base.output)
    output = layers.Dense(230, activation='softmax')(output)
    model = Model(inputs=inputs, outputs=output)
    opti = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=opti, loss=losses.mean_squared_error, metrics=[metrics.categorical_accuracy])

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    output = layers.GlobalMaxPooling2D()(base.output)
    output = layers.Dense(1024, activation='sigmoid')(output)
    middle_layer_model = Model(inputs=model.input, outputs=output)

    return model, middle_layer_model


if __name__ == '__main__':
    # ime_model()
    raw_model()
