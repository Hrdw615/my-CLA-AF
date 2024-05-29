# CLA-AF模型

import warnings
from keras.layers import Conv1D, BatchNormalization, Activation, Dense
from keras.layers import Dropout, Concatenate, Flatten, Lambda
from keras import regularizers
from keras.layers import Reshape, Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.python.keras.layers import RepeatVector, Permute, Multiply, MaxPooling1D, GRU

warnings.filterwarnings("ignore")


class Net(object):
    def __init__(self):
        pass

    @staticmethod
    def __slice(x, index):
        return x[:, :, index]

    @staticmethod
    def __backbone(inp, C=0.001, initial='he_normal'):
        """
        # CNN-BiLSTM对单个片段进行学习
        :param inp:  keras tensor, 输入
        :param C:   double, 正则化系数
        :param initial:  str, 初始化方式
        :return: keras tensor, 输出
        """
        net = Conv1D(4, 31, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(inp)
        net = BatchNormalization()(net)  # 归一化
        net = Activation('relu')(net)  # RELU激活函数
        net = MaxPooling1D(5, 5)(net)  # 最大池化
        # net = nn.AdaptiveAvgPool1d((5, 5))(net)  # 自适应平均池化

        net = Conv1D(8, 11, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = MaxPooling1D(5, 5)(net)
        # net = nn.AdaptiveAvgPool1d((5, 5))(net)

        net = Conv1D(8, 7, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = MaxPooling1D(5, 5)(net)
        # net = nn.AdaptiveAvgPool1d((5, 5))(net)

        net = Conv1D(16, 5, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = MaxPooling1D(int(net.shape[1]), int(net.shape[1]))(net)
        # net = nn.AdaptiveAvgPool1d((int(net.shape[1]), int(net.shape[1])))(net)

        net = Bidirectional(LSTM(1, return_sequences=True), merge_mode='concat')(net)  # BiLSTM
        # net = Bidirectional(GRU(1, return_sequences=True), merge_mode='concat')(net)  # BiGRU
        # net = LSTM(1, return_sequences=True)(net)  # LSTM
        # net = GRU(1, return_sequences=True)(net)  # GRU

        return net

    @staticmethod
    def nnet(inputs, keep_prob, num_classes):
        """
        # 整体CLA-AF模型
        :param inputs: keras tensor, 完整ECG的所有片段
        :param keep_prob: float, 随机片段屏蔽概率
        :param num_classes: int, 数据集的类别数
        :return: keras tensor， 每个类别的概率序列
        """
        branches = []
        for i in range(int(inputs.shape[-1])):
            ld = Lambda(Net.__slice, output_shape=(int(inputs.shape[1]), 1), arguments={'index': i})(inputs)
            ld = Reshape((int(inputs.shape[1]), 1))(ld)
            bch = Net.__backbone(ld)
            branches.append(bch)

        # BiLSTM-Attention
        features = Concatenate(axis=1)(branches)
        features = Dropout(keep_prob, [1, int(inputs.shape[-1]), 1])(features)
        # features = Bidirectional(LSTM(1, return_sequences=True), merge_mode='concat')(features)  # BiLSTM

        # Attention
        attention = Dense(1, activation='tanh')(features)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(2)(attention)
        attention = Permute([2, 1])(attention)

        features = Multiply()([features, attention])
        # features = Lambda(lambda xin: np.sum(xin, axis=-2), output_shape=(2,))(features)

        features = Bidirectional(LSTM(1, return_sequences=True), merge_mode='concat')(features)  # BiLSTM
        # features = Bidirectional(GRU(1, return_sequences=True), merge_mode='concat')(features)  # BiGRU
        # features = LSTM(1, return_sequences=True)(features)  # LSTM
        # features = GRU(1, return_sequences=True)(features)  # GRU

        features = Flatten()(features)
        net = Dense(units=num_classes, activation='softmax')(features)
        return net, features

