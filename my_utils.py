# 相关参数配置和辅助函数

import numpy as np
import pywt
import tensorflow as tf
from itertools import cycle
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve, recall_score
from scipy.signal import butter, filtfilt


# 参数配置
class Config(object):
    def __init__(self):
        # 随机数种子
        self.RANDOM_STATE = 42

        # 导联数目
        self.LEAD_NUM = 1

        # 类别名称
        self.CLASS_NAME = ['Normal', 'AF', 'Other']

        # 模型存放路径
        # self.MODEL_PATH = 'F:/dataset/AF_2017/train/model/'
        # self.MODEL_PATH = 'H:/dataset/model_2/'
        self.MODEL_PATH = 'H:/dataset/try/model_87/'

        # 训练数据存放路径
        self.DATA_PATH = 'F:/dataset/ZZZX_ECG/train/'
        # self.DATA_PATH = 'H:/dataset/CSPC2018/train/'
        # self.DATA_PATH = 'F:/dataset/AF_2017/train/train/'
        # self.DATA_PATH = 'H:/dataset/AF_2017/train/'   # 重采样
        # self.DATA_PATH = 'H:/dataset/PTB/train/'
        # self.DATA_PATH = 'H:/dataset/train/'
        # self.DATA_PATH_1 = 'H:/dataset/AF_2017/train_3/'
        # self.DATA_PATH_3 = 'H:/dataset/YY2023/train_2/'
        # self.DATA_PATH_1 = 'H:/dataset/train_2/'
        # self.DATA_PATH_2 = 'H:/dataset/val_2/'

        # 训练标签存放路径
        self.REVISED_LABEL = 'F:/dataset/ZZZX_ECG/train.npy'
        # self.REVISED_LABEL = 'H:/dataset/CSPC2018/train.npy'
        # self.REVISED_LABEL = 'F:/dataset/AF_2017/train/train.npy'
        # self.REVISED_LABEL = 'H:/dataset/AF_2017/train.npy'   # 重采样
        # self.REVISED_LABEL = 'H:/dataset/PTB/train.npy'
        # self.REVISED_LABEL = 'H:/dataset/train.npy'
        # self.REVISED_LABEL_1 = 'H:/dataset/AF_2017/train_3.npy'
        # self.REVISED_LABEL_3 = 'H:/dataset/YY2023/train_2.npy'
        # self.REVISED_LABEL_1 = 'H:/dataset/train_2.npy'
        # self.REVISED_LABEL_2 = 'H:/dataset/val_2.npy'

        # 测试数据存放路径
        # self.DATA = 'F:/learn/CPSC_Scheme-master/test/'
        # self.DATA = 'F:/dataset/AF_2017/train/test/'
        self.DATA = 'F:/dataset/ZZZX_ECG/test/'
        # self.DATA = 'H:/dataset/CSPC2018/test/'
        # self.DATA = 'H:/dataset/PTB/test/'
        # self.DATA = 'H:/dataset/PTB/N/'
        # self.DATA = 'H:/dataset/CSPC2018/AF/'
        # self.DATA = 'F:/dataset/AF_2017/train/AF/'
        # self.DATA = 'F:/dataset/ZZZX_ECG/AF/npy/'
        # self.DATA = 'H:/dataset/AF_2017/test/'
        # self.DATA = 'H:/dataset/test/'

        # 测试标签存放路径
        # self.LABEL = 'F:/learn/CPSC_Scheme-master/test.npy'
        # self.LABEL = 'F:/dataset/AF_2017/train/test.npy'
        self.LABEL = 'F:/dataset/ZZZX_ECG/test.npy'
        # self.LABEL = 'H:/dataset/CSPC2018/test.npy'
        # self.LABEL = 'H:/dataset/PTB/test.npy'
        # self.LABEL = 'H:/dataset/PTB/N.npy'
        # self.LABEL = 'H:/dataset/CSPC2018/AF.npy'
        # self.LABEL = 'F:/dataset/AF_2017/train/AF.npy'
        # self.LABEL = 'F:/dataset/ZZZX_ECG/AF/AF.npy'
        # self.LABEL = 'H:/dataset/AF_2017/test.npy'
        # self.LABEL = 'H:/dataset/test.npy'

        # 信号采样率
        self.Fs = 500

        # 信号切片数目
        self.SEG_NUM = 24

        # 信号采样点长度
        self.SEG_LENGTH = 1500

    @staticmethod
    # 学习率衰减方案
    def lr_schedule(epoch):
        lr = 0.1
        if 20 <= epoch < 60:
            lr = 0.01
        if epoch >= 60:
            lr = 0.001
        print('Learning rate: ', lr)
        return lr


# 辅助函数

config = Config()


def evaluation(pred_vt, test_y, true_v, pred_v, class_num, LEAD, PATH):
    """
    # 评价指标
    # 混淆矩阵
    # F1值表
    # 精确值表
    # ROC曲线（特异性x，召回率y）
    # PR曲线（召回率x，精确率y）
    # 准确率
    """

    # 存放混淆矩阵，F1值表，精确值表，准确率
    out_file = open(PATH + 'Evalution_' + str(LEAD) + ".txt", "w", newline="")

    # 真实值和预测值
    print('true = ', file=out_file)
    print(' '.join(map(str, true_v)), ' ', file=out_file)
    print('pred = ', file=out_file)
    print(' '.join(map(str, pred_v)), ' ', file=out_file)
    # 真实值和预测值的混淆矩阵
    Conf_Mat_test = confusion_matrix(true_v, pred_v)
    print(Conf_Mat_test, file=out_file)
    # 混淆矩阵图
    # 绘制混淆矩阵图
    fig, ax = plt.subplots()
    ax.imshow(Conf_Mat_test, cmap=plt.cm.Blues)
    # 添加标签和标题
    ax.set_xticks(np.arange(class_num))
    ax.set_yticks(np.arange(class_num))
    ax.set_xticklabels(config.CLASS_NAME, fontdict={'family': 'Times New Roman', 'size': 14}, rotation=45)
    ax.set_yticklabels(config.CLASS_NAME, fontdict={'family': 'Times New Roman', 'size': 14})
    ax.set_xlabel('Predicted Label', fontdict={'family': 'Times New Roman', 'size': 18})
    ax.set_ylabel('True Label', fontdict={'family': 'Times New Roman', 'size': 18})
    # ax.set_title('Confusion Matrix', fontdict={'family': 'Times New Roman', 'size': 14})
    # 添加注释
    thresh = Conf_Mat_test.max() / 2.
    for i in range(class_num):
        for j in range(class_num):
            ax.text(j, i, Conf_Mat_test[i, j],
                    ha='center', va='center',
                    color='white' if Conf_Mat_test[i, j] > thresh else 'black', fontdict={'family': 'Times New Roman',
                                                                                          'size': 14})
    # 显示图像
    fig.tight_layout()
    a = PATH + 'HX_' + str(LEAD)
    fig.savefig(f'{a}.jpg', dpi=1200)
    # F1值表
    F1s_test = []
    Precision_test = []
    Specificity_test = []
    for j in range(class_num):
        f1t = 2 * Conf_Mat_test[j][j] / (np.sum(Conf_Mat_test[j, :]) + np.sum(Conf_Mat_test[:, j]))
        print('| F1-' + config.CLASS_NAME[j] + ': ' + str(f1t) + ' |', file=out_file)
        F1s_test.append(f1t)
    # print('f1: ', f1_score(true_v, pred_v, average=None), file=out_file)
    print('F1-mean: ' + str(np.mean(F1s_test)), file=out_file)
    # 精确值表
    for j in range(class_num):
        pre = Conf_Mat_test[j][j] / np.sum(Conf_Mat_test[:, j])
        print('| P-' + config.CLASS_NAME[j] + ': ' + str(pre) + ' |', file=out_file)
        Precision_test.append(pre)
    # print('precision: ', precision_score(true_v, pred_v, average=None), file=out_file)
    # 特异性表
    for k in range(class_num):
        TN = Conf_Mat_test[0][0] + Conf_Mat_test[1][1] + Conf_Mat_test[2][2] - Conf_Mat_test[k][k]
        TNFP = np.sum(Conf_Mat_test[0, :]) + np.sum(Conf_Mat_test[1, :]) + np.sum(Conf_Mat_test[2, :]) - np.sum(
            Conf_Mat_test[k, :])
        sp = TN / TNFP
        print('| Sp-' + config.CLASS_NAME[k] + ': ' + str(sp) + ' |', file=out_file)
        Specificity_test.append(sp)
    # 准确率
    print('ACC: ', accuracy_score(true_v, pred_v), file=out_file)
    print('recall: ', recall_score(true_v, pred_v, average=None), file=out_file)
    out_file.close()

    # 为每个类别计算ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(class_num), colors):
        fpr[i], tpr[i], _ = roc_curve(test_y[:, i], pred_vt[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.figure()
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        lw = 2
        plt.plot(fpr[i], tpr[i], color=color,
                 lw=lw, label='ROC of class {0} (area = {1:0.2f})'
                              ''.format(config.CLASS_NAME[i], roc_auc[i]))
        # plt.plot([0, 1], [0, 1], color='navy', lw=lw)
        plt.legend(prop={'family': 'Times New Roman', 'size': 14}, loc="lower right")
        plt.xticks(fontproperties='Times New Roman', size=14)
        plt.yticks(fontproperties='Times New Roman', size=14)
        plt.gca().set_aspect(1)
        plt.xlim([0.0, 1.01])
        plt.ylim([0.0, 1.01])
        plt.xlabel('False Positive Rate', fontdict={'family': 'Times New Roman', 'size': 18})
        plt.ylabel('True Positive Rate', fontdict={'family': 'Times New Roman', 'size': 18})
        # plt.title('ROC of class ' + config.CLASS_NAME[i], fontdict={'family': 'Times New Roman', 'size': 14})
        # plt.legend(loc="lower right")
        a = PATH + 'ROC_' + str(LEAD) + '_' + str(i)
        plt.tight_layout()
        plt.savefig(f'{a}.jpg', dpi=1200)

    # 绘制所有ROC曲线
    plt.figure()
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(class_num), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC of class {0} (area = {1:0.2f})'
                       ''.format(config.CLASS_NAME[i], roc_auc[i]))
    # plt.plot([0, 1], [0, 1], lw=lw)
    plt.legend(prop={'family': 'Times New Roman', 'size': 14}, loc="lower right")
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.gca().set_aspect(1)
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate', fontdict={'family': 'Times New Roman', 'size': 18})
    plt.ylabel('True Positive Rate', fontdict={'family': 'Times New Roman', 'size': 18})
    # plt.title('ROC of all classes', fontdict={'family': 'Times New Roman', 'size': 16})
    a = PATH + 'ROC_' + str(LEAD) + '_all'
    plt.tight_layout()
    plt.savefig(f'{a}.jpg', dpi=1200)

    # 为每个类别绘制PR曲线
    fpr = dict()
    tpr = dict()
    for i, color in zip(range(class_num), colors):
        fpr[i], tpr[i], _ = precision_recall_curve(test_y[:, i], pred_vt[:, i])
        plt.figure()
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        lw = 2
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='PR')
        plt.legend(prop={'family': 'Times New Roman', 'size': 14}, loc="lower left")
        plt.xticks(fontproperties='Times New Roman', size=14)
        plt.yticks(fontproperties='Times New Roman', size=14)
        plt.gca().set_aspect(1)
        plt.xlim([0.0, 1.01])
        plt.ylim([0.0, 1.01])
        plt.xlabel('Recall', fontdict={'family': 'Times New Roman', 'size': 18})
        plt.ylabel('Precision', fontdict={'family': 'Times New Roman', 'size': 18})
        # plt.title('PR of class ' + config.CLASS_NAME[i], fontdict={'family': 'Times New Roman', 'size': 14})
        a = PATH + 'PR_' + str(LEAD) + '_' + str(i)
        plt.tight_layout()
        plt.savefig(f'{a}.jpg', dpi=1200)

    # 绘制所有PR曲线
    plt.figure()
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(class_num), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='PR of class {0} ' ''.format(config.CLASS_NAME[i]))
    plt.legend(prop={'family': 'Times New Roman', 'size': 14}, loc="lower left")
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.gca().set_aspect(1)
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.xlabel('Recall', fontdict={'family': 'Times New Roman', 'size': 18})
    plt.ylabel('Precision', fontdict={'family': 'Times New Roman', 'size': 18})
    # plt.title('PR of all classes', fontdict={'family': 'Times New Roman', 'size': 12})
    a = PATH + 'PR_' + str(LEAD) + '_all'
    plt.tight_layout()
    plt.savefig(f'{a}.jpg', dpi=1200)


# 焦点损失函数
def focus_loss(y_true, y_pred):
    alpha = tf.constant(2.0)  # Adjustable hyperparameter
    gamma = tf.constant(2.0)  # Adjustable hyperparameter

    loss = tf.square(y_true - y_pred)
    focusloss = alpha * tf.pow(1 - y_pred, gamma) * loss

    return tf.reduce_mean(focusloss)


# 焦点损失函数
from keras import backend as K
def focal_loss(y_true, y_pred):
    alpha, gamma = 0.25, 2
    y_pred = K.clip(y_pred, 1e-8, 1 - 1e-8)
    return - alpha * y_true * K.log(y_pred) * (1 - y_pred)**gamma\
           - (1 - alpha) * (1 - y_true) * K.log(1 - y_pred) * y_pred**gamma


# 使用小波变换对单导联ECG滤波
def WTfilt_1d(sig):
    """
    :param sig: 1-D numpy Array，单导联ECG
    :return: 1-D numpy Array，滤波后信号
    """
    coeffs = pywt.wavedec(sig, 'db6', level=9)
    coeffs[-1] = np.zeros(len(coeffs[-1]))
    coeffs[-2] = np.zeros(len(coeffs[-2]))
    coeffs[0] = np.zeros(len(coeffs[0]))
    sig_filt = pywt.waverec(coeffs, 'db6')
    return sig_filt


def BLfilt_1d(sig):
    # b, a = signal.butter(8, 0.01, 'highpass')
    # baseline = signal.filtfilt(b, a, sig)
    N = 4  # Filter order
    Wn = 0.11  # Cutoff frequency
    B, A = butter(N, Wn, output='ba')
    smooth_data = filtfilt(B, A, sig)
    return smooth_data


# 对单导联ECG进行分割
def SegSig_1d(sig, seg_length=1500, overlap_length=0,
              full_seg=True, stt=0):
    """
    :param sig: 1-D numpy Array，单导联ECG
    :param seg_length:  int，片段的采样点长度
    :param overlap_length: int, 片段之间相互覆盖的采样点长度，默认为0
    :param full_seg:  bool， 是否对末尾不足seg_length的片段进行延拓并分割，默认True
    :param stt:  int, 开始进行分割的位置， 默认从头开始（0）
    :return: 2-D numpy Array, 片段个数 * 片段长度
    """
    length = len(sig)
    SEGs = np.zeros([1, seg_length])
    start = stt
    while start + seg_length <= length:
        tmp = sig[start:start + seg_length].reshape([1, seg_length])
        SEGs = np.concatenate((SEGs, tmp))
        start += seg_length
        start -= overlap_length
    if full_seg:
        if start < length:
            pad_length = seg_length - (length - start)
            tmp = np.concatenate((sig[start:length].reshape([1, length - start]),
                                  sig[:pad_length].reshape([1, pad_length])), axis=1)
            SEGs = np.concatenate((SEGs, tmp))
    SEGs = SEGs[1:]
    return SEGs


# 对小于target_length的片段进行补零
def Pad_1d(sig, target_length):
    """
    :param sig: 1-D numpy Array，输入信号
    :param target_length: int，目标长度
    :return:  1-D numpy Array，输出补零后的信号
    """
    pad_length = target_length - sig.shape[0]
    if pad_length > 0:
        sig = np.concatenate((sig, np.zeros(int(pad_length))))
    return sig


# 对单导联信号滤波，按照指定切片数目和长度进行切片，并堆叠为矩阵
def Stack_Segs_generate(sig, seg_num=24, seg_length=1500, full_seg=True, stt=0):
    """
    :param sig: 1-D numpy Array, 输入单导联ECG
    :param seg_num: int，指定片段个数
    :param seg_length: int，指定片段采样点长度
    :param full_seg: bool，是否对片段末尾不足seg_length的片段进行延拓并分割，默认True
    :param stt: int, 开始进行分割的位置， 默认从头开始（0）
    :return: 3-D numpy Array, 1 * 切片长度 * 切片个数
    """
    # out_file = open('H:/dataset/Filtering/filter' + ".txt", "w", newline="")
    # print(' '.join(map(str, sig)), ' ', file=out_file)
    sig = WTfilt_1d(sig)
    # print(' '.join(map(str, sig)), ' ', file=out_file)
    if len(sig) < seg_length + seg_num:
        sig = Pad_1d(sig, target_length=(seg_length + seg_num - 1))

    overlap_length = int(seg_length - (len(sig) - seg_length) / (seg_num - 1))

    if (len(sig) - seg_length) % (seg_num - 1) == 0:
        full_seg = False

    SEGs = SegSig_1d(sig, seg_length=seg_length,
                     overlap_length=overlap_length, full_seg=full_seg, stt=stt)
    del sig
    SEGs = SEGs.transpose()
    SEGs = SEGs.reshape([1, SEGs.shape[0], SEGs.shape[1]])
    return SEGs


# 对单导联ECG进行滤波，按照指定数目和长度进行分割，并堆叠为矩阵
def Fetch_Pats_Lbs_sLead(Pat_files, Path, target_lead=1, seg_num=24,
                         seg_length=1500, full_seg=True, stt=0, buf_size=100):
# def Fetch_Pats_Lbs_sLead(Pat_files, Path, seg_num=24,
#                          seg_length=1500, full_seg=True, stt=0, buf_size=100):
    """
    :param Pat_files: list or 1-D numpy Array, 指定文件
    :param Path: str，数据存放路径
    # :param target_lead: int，指定单导联，例如1指II导联
    :param seg_num: int，指定片段个数
    :param seg_length: int，指定片段采样点长度
    :param full_seg: bool，是否对末尾不足seg_length的片段进行延拓并分割，默认True
    :param stt: int, 开始进行分割的位置， 默认从头开始（0）
    :param buf_size: 用于加速过程的缓存Array大小，默认为100
    :return:
    """
    seg_length = int(seg_length)
    SEG_buf = np.zeros([1, seg_length, seg_num])
    SEGs = np.zeros([1, seg_length, seg_num])
    for i in range(len(Pat_files)):
        sig = np.load(Path + Pat_files[i])[target_lead, :]
        # sig = np.load(Path + Pat_files[i])
        SEGt = Stack_Segs_generate(sig, seg_num=seg_num,
                                   seg_length=seg_length, full_seg=full_seg, stt=stt)
        SEG_buf = np.concatenate((SEG_buf, SEGt))
        del SEGt
        if SEG_buf.shape[0] >= buf_size:
            SEGs = np.concatenate((SEGs, SEG_buf[1:]))
            del SEG_buf
            SEG_buf = np.zeros([1, seg_length, seg_num])
    if SEG_buf.shape[0] > 1:
        SEGs = np.concatenate((SEGs, SEG_buf[1:]))
    del SEG_buf
    return SEGs[1:]


# 通过随机过采样使各类样本数目为指定比例
def class_balance(records, labels, rand_seed):
    """
    :param records: 1-D numpy Array，不平衡样本记录名集合
    :param labels: 1-D numpy Array，对应标签
    :param rand_seed：int, 随机数种子
    :return: 平衡后的记录名集合和对应标签
    """
    class_num = len(np.unique(labels))
    num_categories = []
    for i in range(class_num):
        num_categories.append(len(labels[labels == i]))
    rate = 1
    records_this_class = records[labels == 1]
    oversample_size = int(np.ceil(num_categories[1] * rate))
    np.random.seed(rand_seed)
    rand_sample = np.random.choice(records_this_class,
                                   size=oversample_size,
                                   replace=False)
    records = np.concatenate((records, rand_sample))
    labels = np.concatenate((labels, np.ones(oversample_size) * 1))
    return records, labels


# 通过随机过采样使各类样本数目平衡
def oversample_balance(records, labels, rand_seed):
    """
    :param records: 1-D numpy Array，不平衡样本记录名集合
    :param labels: 1-D numpy Array，对应标签
    :param rand_seed：int, 随机数种子
    :return: 平衡后的记录名集合和对应标签
    """
    class_num = len(np.unique(labels))
    num_records = len(records)
    num_categories = []
    for i in range(class_num):
        num_categories.append(len(labels[labels == i]))
    upsample_rate = max(num_categories) / np.array(num_categories) - 1
    for i in range(class_num):
        rate = upsample_rate[i]
        if 1 > rate > 0:
            records_this_class = records[labels == i]
            oversample_size = int(np.ceil(num_categories[i] * rate))
            np.random.seed(rand_seed)
            rand_sample = np.random.choice(records_this_class,
                                           size=oversample_size,
                                           replace=False)
            records = np.concatenate((records, rand_sample))
            labels = np.concatenate((labels, np.ones(oversample_size) * i))
    over_sample_records = []
    over_sample_labels = []
    for i in range(num_records):
        rate = upsample_rate[int(labels[i])]
        if rate >= 1:
            over_sample_records = over_sample_records + [records[i]] * int(round(rate))
            over_sample_labels = over_sample_labels + [labels[i]] * int(round(rate))

    records = np.concatenate((records, np.array(over_sample_records)))
    labels = np.concatenate((labels, np.array(over_sample_labels)))
    return records, labels
