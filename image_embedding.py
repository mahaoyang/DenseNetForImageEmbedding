import keras
from keras import layers
from keras import losses
from keras import metrics
from keras import optimizers
from keras.models import Model

from dsnet import dense_net


def ime_model(lr=0.001, shape=(64, 64, 3)):
    inputs = layers.Input(shape=shape)
    base = dense_net(img_input=inputs, blocks=[8, 12, 24, 16])

    o1 = base[1]
    o1 = layers.Dropout(0.5)(o1)
    o1 = layers.Dense(230, activation='softmax')(o1)

    o2 = base[0]
    o2 = layers.Dropout(0.5)(o2)
    o2 = layers.Dense(330, activation='softmax')(o2)

    # x = base[0]
    # o3 = layers.Dense(300, activation='elu')(x)
    # mg = layers.Concatenate(axis=1)([o1, o2, o3])

    mg = layers.Concatenate(axis=1)([o1, o2])
    o4 = layers.Activation('relu')(mg)
    o4 = layers.BatchNormalization(axis=1, epsilon=1.001e-5)(o4)
    o4 = layers.Dropout(0.5)(o4)
    o4 = layers.Dense(230, activation='softmax')(o4)

    model = Model(inputs=inputs, outputs=[o1, o2, o4])

    opti = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(optimizer=opti,
                  loss=[losses.categorical_crossentropy, losses.categorical_crossentropy,
                        losses.categorical_crossentropy],
                  metrics=[metrics.categorical_accuracy])
    model.summary()
    return model


if __name__ == '__main__':
    ime_model()
