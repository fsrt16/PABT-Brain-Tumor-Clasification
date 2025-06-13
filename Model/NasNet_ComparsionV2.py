import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import NASNetMobile, NASNetLarge, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, LSTM, Conv1D, MaxPooling1D, Reshape, Add, Concatenate, Multiply, BatchNormalization

class NASNetBase:
    def __init__(self, input_shape, num_classes, backbone='NASNetMobile'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.backbone = self.get_backbone(backbone)

    def get_backbone(self, name):
        base_model = None
        if name == 'NASNetMobile':
            base_model = NASNetMobile(include_top=False, input_shape=self.input_shape, weights='imagenet')
        elif name == 'NASNetLarge':
            base_model = NASNetLarge(include_top=False, input_shape=self.input_shape, weights='imagenet')
        elif name == 'EfficientNetB0':
            base_model = EfficientNetB0(include_top=False, input_shape=self.input_shape, weights='imagenet')
        else:
            raise ValueError("Unsupported backbone")

        for layer in base_model.layers:
            layer.trainable = False
        return base_model

    def add_residual_block(self, x, filters):
        shortcut = x
        x = Conv1D(filters, 3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

class NASNetANN(NASNetBase):
    def build_model(self):
        inputs = Input(shape=self.input_shape)
        x = self.backbone(inputs)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(self.num_classes, activation='softmax')(x)
        return Model(inputs, x)

class NASNetDNN(NASNetBase):
    def build_model(self):
        inputs = Input(shape=self.input_shape)
        x = self.backbone(inputs)
        x = GlobalAveragePooling2D()(x)
        for units in [1024, 512, 256]:
            residual = Dense(units)(x)
            x = Dense(units, activation='relu')(x)
            x = Add()([x, residual])
            x = Dropout(0.3)(x)
        x = Dense(self.num_classes, activation='softmax')(x)
        return Model(inputs, x)

class NASNetCNN(NASNetBase):
    def build_model(self):
        inputs = Input(shape=self.input_shape)
        x = self.backbone(inputs)
        x = GlobalAveragePooling2D()(x)
        x = Reshape((x.shape[1], 1))(x)
        x = Conv1D(64, 3, activation='relu', padding='same')(x)
        x = self.add_residual_block(x, 64)
        x = MaxPooling1D(2)(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(self.num_classes, activation='softmax')(x)
        return Model(inputs, x)

class NASNetLSTM(NASNetBase):
    def build_model(self):
        inputs = Input(shape=self.input_shape)
        x = self.backbone(inputs)
        x = GlobalAveragePooling2D()(x)
        x = Reshape((x.shape[1], 1))(x)
        x = LSTM(128, return_sequences=True)(x)
        x = LSTM(64)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(self.num_classes, activation='softmax')(x)
        return Model(inputs, x)

class NASNetCNNLSTM(NASNetBase):
    def build_model(self):
        inputs = Input(shape=self.input_shape)
        x = self.backbone(inputs)
        x = GlobalAveragePooling2D()(x)
        x = Reshape((x.shape[1], 1))(x)
        x = Conv1D(64, 3, activation='relu', padding='same')(x)
        x = self.add_residual_block(x, 64)
        x = LSTM(64)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(self.num_classes, activation='softmax')(x)
        return Model(inputs, x)

class NASNetCACNNDNN(NASNetBase):
    def se_block(self, input_tensor, ratio=8):
        filters = input_tensor.shape[-1]
        se = GlobalAveragePooling2D()(input_tensor)
        se = Dense(filters // ratio, activation='relu')(se)
        se = Dense(filters, activation='sigmoid')(se)
        return Multiply()([input_tensor, se])

    def build_model(self):
        inputs = Input(shape=self.input_shape)
        x = self.backbone(inputs)
        x = self.se_block(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        res = x
        x = Dense(512, activation='relu')(x)
        x = Add()([x, res])
        x = Dropout(0.3)(x)
        x = Dense(self.num_classes, activation='softmax')(x)
        return Model(inputs, x)

# Example usage:
# model = NASNetCNNLSTM((224, 224, 3), 3, backbone='NASNetLarge').build_model()
# model.summary()
