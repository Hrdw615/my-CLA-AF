# 训练模型

import os
import warnings
import numpy as np
import my_utils
import tensorflow as tf
from keras import optimizers
from keras.layers import Input
from keras.models import Model, load_model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from my_model import Net
from my_utils import Config, focal_loss

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
warnings.filterwarnings("ignore")
config = Config()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

records_name = np.array(os.listdir(config.DATA_PATH))
records_label = np.load(config.REVISED_LABEL) - 1
class_num = len(np.unique(records_label))

# 划分训练和验证集
train_records, val_records, train_labels, val_labels = train_test_split(
    records_name, records_label, test_size=0.2, random_state=config.RANDOM_STATE)
num_categories = []
for i in range(class_num):
    num_categories.append(len(train_labels[train_labels == i]))
print(num_categories)

# 过采样使训练和验证集样本分布平衡
# 1:1:1
# train_records, train_labels = utils.oversample_balance(train_records, train_labels, config.RANDOM_STATE)
# num_categories = []
# for i in range(class_num):
#     num_categories.append(len(train_labels[train_labels == i]))
# print(num_categories)
# 1:2:1
# train_records, train_labels = utils.class_balance(train_records, train_labels, config.RANDOM_STATE)
# # val_records, val_labels = utils.oversample_balance(val_records, val_labels, config.RANDOM_STATE)
# num_categories = []
# for i in range(class_num):
#     num_categories.append(len(train_labels[train_labels == i]))
# print(num_categories)

# # 先平衡后合并
# # 第一个数据集
# records_name_1 = np.array(os.listdir(config.DATA_PATH_1))
# records_label_1 = np.load(config.REVISED_LABEL_1) - 1
# class_num = len(np.unique(records_label_1))
#
# # 划分训练，验证集
# train_records_1, val_records_1, train_labels_1, val_labels_1 = train_test_split(
#     records_name_1, records_label_1, test_size=0.2, random_state=config.RANDOM_STATE)
# num_categories_1 = []
# for i in range(class_num):
#     num_categories_1.append(len(train_labels_1[train_labels_1 == i]))
# print(num_categories_1)
#
# # 过采样使训练和验证集样本分布平衡
# # 1:1:1
# train_records_1, train_labels_1 = utils.oversample_balance(train_records_1, train_labels_1, config.RANDOM_STATE)
# num_categories_1 = []
# for i in range(class_num):
#     num_categories_1.append(len(train_labels_1[train_labels_1 == i]))
# print(num_categories_1)
#
# # 第二个数据集
# records_name_2 = np.array(os.listdir(config.DATA_PATH_2))
# records_label_2 = np.load(config.REVISED_LABEL_2) - 1
#
# # 划分训练，验证集
# train_records_2, val_records_2, train_labels_2, val_labels_2 = train_test_split(
#     records_name_2, records_label_2, test_size=0.2, random_state=config.RANDOM_STATE)
# num_categories_2 = []
# for i in range(class_num):
#     num_categories_2.append(len(train_labels_2[train_labels_2 == i]))
# print(num_categories_2)
#
# # 过采样使训练和验证集样本分布平衡
# # 1:1:1
# train_records_2, train_labels_2 = utils.oversample_balance(train_records_2, train_labels_2, config.RANDOM_STATE)
# num_categories_2 = []
# for i in range(class_num):
#     num_categories_2.append(len(train_labels_2[train_labels_2 == i]))
# print(num_categories_2)
#
# # 第三个数据集
# records_name_3 = np.array(os.listdir(config.DATA_PATH_3))
# records_label_3 = np.load(config.REVISED_LABEL_3) - 1
#
# # 划分训练，验证集
# train_records_3, val_records_3, train_labels_3, val_labels_3 = train_test_split(
#     records_name_3, records_label_3, test_size=0.2, random_state=config.RANDOM_STATE)
# num_categories_3 = []
# for i in range(class_num):
#     num_categories_3.append(len(train_labels_3[train_labels_3 == i]))
# print(num_categories_3)
#
# # 过采样使训练和验证集样本分布平衡
# # 1:1:1
# train_records_3, train_labels_3 = utils.oversample_balance(train_records_3, train_labels_3, config.RANDOM_STATE)
# num_categories_3 = []
# for i in range(class_num):
#     num_categories_3.append(len(train_labels_3[train_labels_3 == i]))
# print(num_categories_3)
#
# # 第四个数据集
# records_name_4 = np.array(os.listdir(config.DATA_PATH_4))
# records_label_4 = np.load(config.REVISED_LABEL_4) - 1
#
# # 划分训练，验证集
# train_records_4, val_records_4, train_labels_4, val_labels_4 = train_test_split(
#     records_name_4, records_label_4, test_size=0.2, random_state=config.RANDOM_STATE)
# num_categories_4 = []
# for i in range(class_num):
#     num_categories_4.append(len(train_labels_4[train_labels_4 == i]))
# print(num_categories_4)
#
# # 过采样使训练和验证集样本分布平衡
# # 1:1:1
# train_records_4, train_labels_4 = utils.oversample_balance(train_records_4, train_labels_4, config.RANDOM_STATE)
# num_categories_4 = []
# for i in range(class_num):
#     num_categories_4.append(len(train_labels_4[train_labels_4 == i]))
# print(num_categories_4)

# train_records = np.array(os.listdir(config.DATA_PATH_1))
# train_labels = np.load(config.REVISED_LABEL_1) - 1
# class_num = len(np.unique(train_labels))
# val_records = np.array(os.listdir(config.DATA_PATH_2))
# val_labels = np.load(config.REVISED_LABEL_2) - 1

PATH = 'H:/dataset/try/eva_87/'

# 读取数据并进行分割和预处理
TARGET_LEAD = 1
# train_x = utils.Fetch_Pats_Lbs_sLead(train_records, Path=config.DATA_PATH_1,
#                                      seg_num=config.SEG_NUM,
#                                      seg_length=config.SEG_LENGTH)
# train_y = to_categorical(train_labels, num_classes=class_num)
# val_x = utils.Fetch_Pats_Lbs_sLead(val_records, Path=config.DATA_PATH_2,
#                                    seg_num=config.SEG_NUM,
#                                    seg_length=config.SEG_LENGTH)
# val_y = to_categorical(val_labels, num_classes=class_num)
train_x = my_utils.Fetch_Pats_Lbs_sLead(train_records, Path=config.DATA_PATH,
                                        target_lead=TARGET_LEAD, seg_num=config.SEG_NUM,
                                        seg_length=config.SEG_LENGTH)
train_y = to_categorical(train_labels, num_classes=class_num)
val_x = my_utils.Fetch_Pats_Lbs_sLead(val_records, Path=config.DATA_PATH,
                                      target_lead=TARGET_LEAD, seg_num=config.SEG_NUM,
                                      seg_length=config.SEG_LENGTH)
val_y = to_categorical(val_labels, num_classes=class_num)
del train_records, train_labels
del val_records, val_labels

# train_x_2 = utils.Fetch_Pats_Lbs_sLead(train_records_2, Path=config.DATA_PATH_2,
#                                        seg_num=config.SEG_NUM,
#                                        seg_length=config.SEG_LENGTH)
# train_y_2 = to_categorical(train_labels_2, num_classes=class_num)
# val_x_2 = utils.Fetch_Pats_Lbs_sLead(val_records_2, Path=config.DATA_PATH_2,
#                                      seg_num=config.SEG_NUM,
#                                      seg_length=config.SEG_LENGTH)
# val_y_2 = to_categorical(val_labels_2, num_classes=class_num)
#
# train_x_3 = utils.Fetch_Pats_Lbs_sLead(train_records_3, Path=config.DATA_PATH_3,
#                                        seg_num=config.SEG_NUM,
#                                        seg_length=config.SEG_LENGTH)
# train_y_3 = to_categorical(train_labels_3, num_classes=class_num)
# val_x_3 = utils.Fetch_Pats_Lbs_sLead(val_records_3, Path=config.DATA_PATH_3,
#                                      seg_num=config.SEG_NUM,
#                                      seg_length=config.SEG_LENGTH)
# val_y_3 = to_categorical(val_labels_3, num_classes=class_num)
#
# train_x_4 = utils.Fetch_Pats_Lbs_sLead(train_records_4, Path=config.DATA_PATH_4,
#                                        seg_num=config.SEG_NUM,
#                                        seg_length=config.SEG_LENGTH)
# train_y_4 = to_categorical(train_labels_4, num_classes=class_num)
# val_x_4 = utils.Fetch_Pats_Lbs_sLead(val_records_4, Path=config.DATA_PATH_4,
#                                      seg_num=config.SEG_NUM,
#                                      seg_length=config.SEG_LENGTH)
# val_y_4 = to_categorical(val_labels_4, num_classes=class_num)
#
# del train_records_1, train_labels_1
# del val_records_1, val_labels_1
# del train_records_2, train_labels_2
# del val_records_2, val_labels_2
# del train_records_3, train_labels_3
# del val_records_3, val_labels_3
# del train_records_4, train_labels_4
# del val_records_4, val_labels_4
#
# # 合并
# train_x = np.concatenate((train_x_1, train_x_2, train_x_3, train_x_4))
# del train_x_1, train_x_2, train_x_3, train_x_4
# train_y = np.concatenate((train_y_1, train_y_2, train_y_3, train_y_4))
# del train_y_1, train_y_2, train_y_3, train_y_4
# val_x = np.concatenate((val_x_1, val_x_2, val_x_3, val_x_4))
# del val_x_1, val_x_2, val_x_3, val_x_4
# val_y = np.concatenate((val_y_1, val_y_2, val_y_3, val_y_4))
# del val_y_1, val_y_2, val_y_3, val_y_4

model_name = 'net_lead_' + str(TARGET_LEAD) + '.hdf5'

print('Scaling data ...-----------------\n')
for j in range(train_x.shape[0]):
    train_x[j, :, :] = scale(train_x[j, :, :], axis=0)
for j in range(val_x.shape[0]):
    val_x[j, :, :] = scale(val_x[j, :, :], axis=0)

# 设置超参数，训练模型
batch_size = 64
epochs = 100
momentum = 0.9
keep_prob = 0.5

inputs = Input(shape=(config.SEG_LENGTH, config.SEG_NUM))  # 输入层
net = Net()
outputs, _ = net.nnet(inputs, keep_prob, num_classes=class_num)
model = Model(inputs=inputs, outputs=outputs)

opt = optimizers.SGD(lr=config.lr_schedule(0), momentum=momentum)
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])
# model.compile(optimizer=opt, loss=utils.focus_loss,
#               metrics=['categorical_accuracy'])

checkpoint = ModelCheckpoint(filepath=config.MODEL_PATH + model_name,
                             monitor='val_categorical_accuracy', mode='max',
                             save_best_only='True')
lr_scheduler = LearningRateScheduler(config.lr_schedule)
# lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='min')
# lr_scheduler = CosineAnnealingLR(optimizer, T_max=80, eta_min=0.0000001)
callback_lists = [checkpoint, lr_scheduler]
model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, verbose=2,
          validation_data=(val_x, val_y), callbacks=callback_lists)  # 训练

del train_x, train_y

model = load_model(config.MODEL_PATH + model_name)
# model = load_model(config.MODEL_PATH + model_name, custom_objects={'focal_loss': utils.focal_loss})

pred_vt = model.predict(val_x, batch_size=batch_size, verbose=1)
pred_v = np.argmax(pred_vt, axis=1)
true_v = np.argmax(val_y, axis=1)
my_utils.evaluation(pred_vt, val_y, true_v, pred_v, class_num, TARGET_LEAD, PATH)
del val_x, val_y

# 评估模型在验证集上的性能
print('\nResult\n')
Conf_Mat_val = confusion_matrix(true_v, pred_v)  # 真实值和预测值的混淆矩阵
print(Conf_Mat_val)
F1s_val = []
for j in range(class_num):
    f1t = 2 * Conf_Mat_val[j][j] / (np.sum(Conf_Mat_val[j, :]) + np.sum(Conf_Mat_val[:, j]))
    print('| F1-' + config.CLASS_NAME[j] + ':' + str(f1t) + ' |')
    F1s_val.append(f1t)
print('F1-mean: ' + str(np.mean(F1s_val)))
for j in range(class_num):
    acc = Conf_Mat_val[j][j] / np.sum(Conf_Mat_val[:, j])
    print('| P-' + config.CLASS_NAME[j] + ':' + str(acc) + ' |')
print('ACC: ', accuracy_score(true_v, pred_v))

# # 先平衡后合并
# # 第一个数据集
# records_name_1 = np.array(os.listdir(config.DATA_PATH_1))
# records_label_1 = np.load(config.REVISED_LABEL_1) - 1
# class_num = len(np.unique(records_label_1))
#
# # 划分训练，验证集
# train_records_1, val_records_1, train_labels_1, val_labels_1 = train_test_split(
#     records_name_1, records_label_1, test_size=0.2, random_state=config.RANDOM_STATE)
# num_categories_1 = []
# for i in range(class_num):
#     num_categories_1.append(len(train_labels_1[train_labels_1 == i]))
# print(num_categories_1)
#
# # 过采样使训练和验证集样本分布平衡
# # 1:1:1
# train_records_1, train_labels_1 = utils.oversample_balance(train_records_1, train_labels_1, config.RANDOM_STATE)
# num_categories_1 = []
# for i in range(class_num):
#     num_categories_1.append(len(train_labels_1[train_labels_1 == i]))
# print(num_categories_1)
#
# # 第二个数据集
# records_name_2 = np.array(os.listdir(config.DATA_PATH_2))
# records_label_2 = np.load(config.REVISED_LABEL_2) - 1
#
# # 划分训练，验证集
# train_records_2, val_records_2, train_labels_2, val_labels_2 = train_test_split(
#     records_name_2, records_label_2, test_size=0.2, random_state=config.RANDOM_STATE)
# num_categories_2 = []
# for i in range(class_num):
#     num_categories_2.append(len(train_labels_2[train_labels_2 == i]))
# print(num_categories_2)
#
# # 过采样使训练和验证集样本分布平衡
# # 1:1:1
# train_records_2, train_labels_2 = utils.oversample_balance(train_records_2, train_labels_2, config.RANDOM_STATE)
# num_categories_2 = []
# for i in range(class_num):
#     num_categories_2.append(len(train_labels_2[train_labels_2 == i]))
# print(num_categories_2)
#
# # 第三个数据集
# records_name_3 = np.array(os.listdir(config.DATA_PATH_3))
# records_label_3 = np.load(config.REVISED_LABEL_3) - 1
#
# # 划分训练，验证集
# train_records_3, val_records_3, train_labels_3, val_labels_3 = train_test_split(
#     records_name_3, records_label_3, test_size=0.2, random_state=config.RANDOM_STATE)
# num_categories_3 = []
# for i in range(class_num):
#     num_categories_3.append(len(train_labels_3[train_labels_3 == i]))
# print(num_categories_3)
#
# # 过采样使训练和验证集样本分布平衡
# # 1:1:1
# train_records_3, train_labels_3 = utils.oversample_balance(train_records_3, train_labels_3, config.RANDOM_STATE)
# num_categories_3 = []
# for i in range(class_num):
#     num_categories_3.append(len(train_labels_3[train_labels_3 == i]))
# print(num_categories_3)
#
# # # 合并数据集
# # train_records = np.concatenate((train_records_1, train_records_2, train_records_3))
# # train_labels = np.concatenate((train_labels_1, train_labels_2, train_labels_3))
# # val_records = np.concatenate((val_records_1, val_records_2, val_records_3))
# # val_labels = np.concatenate((val_labels_1, val_labels_2, val_labels_3))
# # num_categories = []
# # for i in range(class_num):
# #     num_categories.append(len(train_labels[train_labels == i]))
# # print(num_categories)
#
# PATH = 'H:/dataset/try/eva_32/'
#
# # 取出训练集和测试集病人对应导联信号，并进行切片和z-score标准化
# print('Fetching data ...-----------------\n')
# TARGET_LEAD = 1
# train_x_1 = utils.Fetch_Pats_Lbs_sLead(train_records_1, Path=config.DATA_PATH_1,
#                                        target_lead=TARGET_LEAD, seg_num=config.SEG_NUM,
#                                        seg_length=config.SEG_LENGTH)
# train_y_1 = to_categorical(train_labels_1, num_classes=class_num)
# val_x_1 = utils.Fetch_Pats_Lbs_sLead(val_records_1, Path=config.DATA_PATH_1,
#                                      target_lead=TARGET_LEAD, seg_num=config.SEG_NUM,
#                                      seg_length=config.SEG_LENGTH)
# val_y_1 = to_categorical(val_labels_1, num_classes=class_num)
#
# train_x_2 = utils.Fetch_Pats_Lbs_sLead(train_records_2, Path=config.DATA_PATH_2,
#                                        target_lead=TARGET_LEAD, seg_num=config.SEG_NUM,
#                                        seg_length=config.SEG_LENGTH)
# train_y_2 = to_categorical(train_labels_2, num_classes=class_num)
# val_x_2 = utils.Fetch_Pats_Lbs_sLead(val_records_2, Path=config.DATA_PATH_2,
#                                      target_lead=TARGET_LEAD, seg_num=config.SEG_NUM,
#                                      seg_length=config.SEG_LENGTH)
# val_y_2 = to_categorical(val_labels_2, num_classes=class_num)
#
# train_x_3 = utils.Fetch_Pats_Lbs_sLead(train_records_3, Path=config.DATA_PATH_3,
#                                        target_lead=TARGET_LEAD, seg_num=config.SEG_NUM,
#                                        seg_length=config.SEG_LENGTH)
# train_y_3 = to_categorical(train_labels_3, num_classes=class_num)
# val_x_3 = utils.Fetch_Pats_Lbs_sLead(val_records_3, Path=config.DATA_PATH_3,
#                                      target_lead=TARGET_LEAD, seg_num=config.SEG_NUM,
#                                      seg_length=config.SEG_LENGTH)
# val_y_3 = to_categorical(val_labels_3, num_classes=class_num)
#
# # 合并
# train_x = np.concatenate((train_x_1, train_x_2, train_x_3))
# train_y = np.concatenate((train_y_1, train_y_2, train_y_3))
# val_x = np.concatenate((val_x_1, val_x_2, val_x_3))
# val_y = np.concatenate((val_y_1, val_y_2, val_y_3))
