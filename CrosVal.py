# 五折交叉验证

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
from sklearn.model_selection import train_test_split, KFold
from my_model import Net
from my_utils import Config

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

# 划分训练，验证集 -----------------------------------------------------------------------------------------------
train_records, val_records, train_labels, val_labels = train_test_split(
    records_name, records_label, test_size=0.2, random_state=config.RANDOM_STATE)
num_categories = []
for i in range(class_num):
    num_categories.append(len(train_labels[train_labels == i]))
print(num_categories)

# stratifiedkf = StratifiedKFold(n_splits=5)   # 分层交叉验证
stratifiedkf = KFold(n_splits=5)   # 交叉验证
# accuracy_score_list, recall_score_list, f1_score_list = [], [], []
i = 1
for train_index, test_index in stratifiedkf.split(records_name, records_label):
    # 准备交叉验证的数据
    train_records = records_name[train_index]
    train_labels = records_label[train_index]
    val_records = records_name[test_index]
    val_labels = records_label[test_index]

    PATH = 'H:/dataset/try/eva_77/'

    # 取出训练集和测试集病人对应导联信号，并进行切片和z-score标准化 --------------------------------------------------------
    print('Fetching data ...-----------------\n')
    TARGET_LEAD = 1

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

    model_name = 'net_lead_' + str(TARGET_LEAD) + '_' + str(i) + '.hdf5'

    print('Scaling data ...-----------------\n')
    for j in range(train_x.shape[0]):
        train_x[j, :, :] = scale(train_x[j, :, :], axis=0)
    for j in range(val_x.shape[0]):
        val_x[j, :, :] = scale(val_x[j, :, :], axis=0)

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

    checkpoint = ModelCheckpoint(filepath=config.MODEL_PATH + model_name,
                                 monitor='val_categorical_accuracy', mode='max',
                                 save_best_only='True')
    lr_scheduler = LearningRateScheduler(config.lr_schedule)
    callback_lists = [checkpoint, lr_scheduler]
    model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, verbose=2,
              validation_data=(val_x, val_y), callbacks=callback_lists)  # 训练

    del train_x, train_y

    model = load_model(config.MODEL_PATH + model_name)

    pred_vt = model.predict(val_x, batch_size=batch_size, verbose=1)
    pred_v = np.argmax(pred_vt, axis=1)
    true_v = np.argmax(val_y, axis=1)
    t = str(TARGET_LEAD) + '_' + str(i)
    my_utils.evaluation(pred_vt, val_y, true_v, pred_v, class_num, t, PATH)
    del val_x, val_y

    # 评估模型在验证集上的性能 ---------------------------------------------------------------------------------------------
    print('\nResult for Lead ' + str(TARGET_LEAD) + '_' + str(i) + '-----------------------------\n')
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
    i = i+1

