import keras
from keras import layers
from keras import losses
from keras import metrics
from keras import optimizers
from keras.models import Model

from dsnet import dense_net


def ime_model(lr=0.001, shape=(64, 64, 3)):
    inputs = layers.Input(shape=shape)
    base = dense_net(img_input=inputs, blocks=[4, 8, 16, 16, 8])

    o1 = base[1]
    o1 = layers.Dense(30, activation='sigmoid')(o1)
    o2 = base[2]
    o2 = layers.Dense(300, activation='elu')(o2)
    x = base[0]
    o3 = layers.Dense(230, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=[o1, o2, o3])

    opti = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(optimizer=opti,
                  loss=[losses.mean_squared_error, losses.categorical_crossentropy, losses.categorical_crossentropy],
                  metrics=[metrics.categorical_accuracy])
    model.summary()
