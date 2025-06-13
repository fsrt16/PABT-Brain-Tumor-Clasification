# nasnet_models.py

import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Dropout, Flatten, Conv1D, MaxPooling1D, GlobalAveragePooling2D,
                                     LSTM, Reshape, Concatenate, BatchNormalization, Activation, Multiply)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import NASNetMobile


def channel_attention(input_feature, ratio=8):
    channel_axis = -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = tf.keras.layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = tf.keras.layers.GlobalMaxPooling2D()(input_feature)
    max_pool = tf.keras.layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = tf.keras.layers.Add()([avg_pool, max_pool])
    cbam_feature = tf.keras.layers.Activation('sigmoid')(cbam_feature)

    return tf.keras.layers.Multiply()([input_feature, cbam_feature])


def base_nasnet_model(input_shape):
    base_model = NASNetMobile(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    return base_model.input, x


def nasnet_ann(input_shape, num_classes):
    inputs, base_output = base_nasnet_model(input_shape)
    x = Dense(128, activation='relu')(base_output)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='NASNet_ANN')


def nasnet_dnn(input_shape, num_classes):
    inputs, base_output = base_nasnet_model(input_shape)
    x = Dense(512, activation='relu')(base_output)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='NASNet_DNN')


def nasnet_cnn(input_shape, num_classes):
    inputs, base_output = base_nasnet_model(input_shape)
    x = Reshape((1, -1))(base_output)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='NASNet_CNN')


def nasnet_lstm(input_shape, num_classes):
    inputs, base_output = base_nasnet_model(input_shape)
    x = Reshape((1, -1))(base_output)
    x = LSTM(128, return_sequences=False)(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='NASNet_LSTM')


def nasnet_cnn_lstm(input_shape, num_classes):
    inputs, base_output = base_nasnet_model(input_shape)
    x = Reshape((1, -1))(base_output)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(64)(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='NASNet_CNN_LSTM')


def nasnet_ca_cnn_dnn(input_shape, num_classes):
    inputs, base_output = base_nasnet_model(input_shape)
    x = tf.keras.layers.Reshape((7, 7, -1))(tf.keras.layers.Dense(49 * 32)(base_output))
    x = channel_attention(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='NASNet_CA_CNN_DNN')


# Utility to build all models for summary and comparison
def build_all_models(input_shape, num_classes):
    models = {}
    models['ann'] = nasnet_ann(input_shape, num_classes)
    models['dnn'] = nasnet_dnn(input_shape, num_classes)
    models['cnn'] = nasnet_cnn(input_shape, num_classes)
    models['lstm'] = nasnet_lstm(input_shape, num_classes)
    models['cnn_lstm'] = nasnet_cnn_lstm(input_shape, num_classes)
    models['ca_cnn_dnn'] = nasnet_ca_cnn_dnn(input_shape, num_classes)
    return models


if __name__ == '__main__':
    shape = (224, 224, 3)
    classes = 3
    nets = build_all_models(shape, classes)

    for name, model in nets.items():
        print(f"\nSummary of {model.name}:")
        model.summary()
        print("=" * 80)
