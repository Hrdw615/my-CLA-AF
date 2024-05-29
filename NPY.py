# # 批量改名
# import os
# # import numpy as np
# txt_filepath = "H:/dataset/PTB/test_2/"
# test_filepath = "H:/dataset/PTB/tmp2/"
# file_name_list = os.listdir(txt_filepath)
# rename_format = '{}{}'
# for txt_name in file_name_list:
#     doc_src = os.path.join(txt_filepath, txt_name)
#     name, ext = os.path.splitext(os.path.basename(txt_name))
#     begin_num = 'AAA' + name
#     doc_name = rename_format.format(begin_num, os.path.splitext(txt_name)[-1])
#     doc_dst = os.path.join(test_filepath, doc_name)
#     os.rename(doc_src, doc_dst)


# # 转置txt
# import os
# txt_filepath = "H:/dataset/archive/txt/"
# test_filepath = "H:/dataset/archive/trans/"
# file_name_list = os.listdir(txt_filepath)
# for txt_name in file_name_list:
#     test_name = txt_name.split(".")[0] + ".txt"
#     in_file = open(txt_filepath + txt_name.split(".")[0] + ".txt", "r", newline="")
#     out_file = open(test_filepath + txt_name.split(".")[0] + ".txt", "w", newline="")
#     dict1 = dict()
#     lines = in_file.readlines()  # 读取第一行
#     length = len(lines[0].strip().split())  # 读取一行数据量（列数），即转置后的行数
#     for i in range(length):
#         dict1[i] = []  # 每行数据初始化
#     for line in lines:
#         line = line.strip().split()  # 对每行的每个数据进行划分
#         for j in range(length):
#             # line[j] = str(int(line[j])/5)
#             dict1[j].append(line[j])  # 转置
#     for i in dict1:
#         print(" ".join(dict1[i]), file=out_file)  # 将转置后的每行打印到out_file
#     in_file.close()
#     out_file.close()
#     print("保存文件成功")


# # 批量归一化
# import numpy as np
# import os
#
#
# def z_score_normalize(data):
#     mean = np.mean(data)
#     std = np.std(data)
#     normalized_data = (data - mean) / std
#     return normalized_data
#
#
# def mean_normalize(data):
#     mean_val = np.mean(data)
#     min_val = np.min(data)
#     max_val = np.max(data)
#     normalized_data = (data - mean_val) / (max_val - min_val)
#     return normalized_data
#
#
# txt_filepath = "H:/dataset/archive/trans/"
# test_filepath = "H:/dataset/archive/nor/"
# file_name_list = os.listdir(txt_filepath)
# for txt_name in file_name_list:
#     test_name = txt_name.split(".")[0] + ".txt"
#     in_file = open(txt_filepath + txt_name.split(".")[0] + ".txt", "r", newline="")
#     out_file = open(test_filepath + txt_name.split(".")[0] + ".txt", "w", newline="")
#     lines = in_file.readlines()  # 读取第一行
#     length = len(lines[0].strip().split())
#     data = []
#     for line in lines:
#         data = line.strip().split()  # 对每行的每个数据进行划分
#         data = np.array(data, dtype=np.float64) / 255.0
#         # data = np.array(data, dtype=np.float64)
#         data2 = mean_normalize(data)
#         print(' '.join(map(str, data2)), ' ', file=out_file)
#         # print(' '.join(map(str, data2)), ' ', file=out_file)
#         # print(' '.join(map(str, data2)), ' ', file=out_file)
#         # print(' '.join(map(str, data2)), ' ', file=out_file)
#         # print(' '.join(map(str, data2)), ' ', file=out_file)
#         # print(' '.join(map(str, data2)), ' ', file=out_file)
#         # print(' '.join(map(str, data2)), ' ', file=out_file)
#         # print(' '.join(map(str, data2)), ' ', file=out_file)
#         # print(' '.join(map(str, data2)), ' ', file=out_file)
#         # print(' '.join(map(str, data2)), ' ', file=out_file)
#         # print(' '.join(map(str, data2)), ' ', file=out_file)
#         # print(' '.join(map(str, data2)), ' ', file=out_file)
#     in_file.close()
#     out_file.close()
#     print("保存文件成功")


# # 批量txt转npy
# import os
# import numpy as np
# txt_filepath = "F:/dataset/AF_2017/train/test/"
# file_name_list = os.listdir(txt_filepath)
# for txt_name in file_name_list:
#     name, ext = os.path.splitext(os.path.basename(txt_name))
#     a = 'F:/dataset/AF_2017/train/test/'+name
#     data = np.load(f'{a}.npy')
#     b = 'H:/dataset/AF_2017/test_3/'+name
#     np.save(f'{b}.npy', data[1, :])


# # npy转txt
# import numpy as np
# file = np.load('H:/dataset/PTB/npy/A0003.npy')
# np.savetxt('H:/dataset/PTB/A0003.txt', file)


# # 划分训练，验证与测试集
# import os
# import numpy as np
# from CPSC_config import Config
# from sklearn.model_selection import train_test_split
#
# config = Config()
# records_name = np.array(os.listdir(config.DATA_PATH))
# records_label = np.load(config.REVISED_LABEL) - 1
# class_num = len(np.unique(records_label))
# out_file_train = open('H:/dataset/CSPC2018/' + "train_2.txt", "w", newline="")
# out_file_test = open('H:/dataset/CSPC2018/' + "test_2.txt", "w", newline="")
# train_val_records, test_records, train_val_labels, test_labels = train_test_split(
#     records_name, records_label, test_size=0.2, random_state=config.RANDOM_STATE)
# len_train = len(train_val_records)
# len_test = len(test_records)
# for i in range(len_train):
#     name_train, ext = os.path.splitext(os.path.basename(train_val_records[i]))
#     a = 'H:/dataset/CSPC2018/npy_2/' + name_train
#     data = np.load(f'{a}.npy')
#     if i+1 < 10:
#         b = 'H:/dataset/CSPC2018/train_2/' + 'A000' + str(i+1)
#     elif 10 <= i+1 < 100:
#         b = 'H:/dataset/CSPC2018/train_2/' + 'A00' + str(i+1)
#     elif 100 <= i+1 < 1000:
#         b = 'H:/dataset/CSPC2018/train_2/' + 'A0' + str(i+1)
#     else:
#         b = 'H:/dataset/CSPC2018/train_2/' + 'A' + str(i+1)
#     np.save(f'{b}.npy', data)
#     print(train_val_labels[i]+1, file=out_file_train)
# for j in range(len_test):
#     name_test, ext = os.path.splitext(os.path.basename(test_records[j]))
#     a = 'H:/dataset/CSPC2018/npy_2/' + name_test
#     data = np.load(f'{a}.npy')
#     if j+1 < 10:
#         b = 'H:/dataset/CSPC2018/test_2/' + 'A000' + str(j+1)
#     elif 10 <= j+1 < 100:
#         b = 'H:/dataset/CSPC2018/test_2/' + 'A00' + str(j+1)
#     elif 100 <= j+1 < 1000:
#         b = 'H:/dataset/CSPC2018/test_2/' + 'A0' + str(j+1)
#     else:
#         b = 'H:/dataset/CSPC2018/test_2/' + 'A' + str(j+1)
#     np.save(f'{b}.npy', data)
#     print(test_labels[j]+1, file=out_file_test)
#
# out_file_train.close()
# out_file_test.close()


# # txt转npy
# import numpy as np
# data = np.loadtxt("H:/dataset/val_2.txt")
# np.save("H:/dataset/val_2.npy", data)


# # npy转txt
# import numpy as np
# file = np.load('H:/dataset/AF_2017/test_2.npy')
# np.savetxt('H:/dataset/AF_2017/test_2.txt', file)


# # 批量txt转npy
# import os
# import numpy as np
# txt_filepath = "H:/dataset/AF_2017/nor_2/"
# file_name_list = os.listdir(txt_filepath)
# for txt_name in file_name_list:
#     name, ext = os.path.splitext(os.path.basename(txt_name))
#     a = 'H:/dataset/AF_2017/nor_2/'+name
#     data = np.loadtxt(f'{a}.txt')
#     b = 'H:/dataset/AF_2017/npy_2/'+name
#     np.save(f'{b}.npy', data)


# # 批量txt转npy
# import os
# import numpy as np
# txt_filepath = "H:/dataset/train_4/"
# test_filepath = "H:/dataset/train_32/"
# file_name_list = os.listdir(txt_filepath)
# rename_format = '{}{}'
# # begin_num = 1
# for txt_name in file_name_list:
#     name, ext = os.path.splitext(os.path.basename(txt_name))
#     doc_src = os.path.join(txt_filepath, txt_name)
#     # begin_num = int(name)
#     # name = str(begin_num)
#     # if begin_num < 10:
#     #     name = 'A000' + name
#     # elif 10 <= begin_num < 100:
#     #     name = 'A00' + name
#     # elif 100 <= begin_num < 1000:
#     #     name = 'A0' + name
#     # else:
#     #     name = 'A' + name
#     name = 'A' + name
#     # begin_num += 1
#     doc_name = rename_format.format(name, os.path.splitext(txt_name)[-1])
#     doc_dst = os.path.join(test_filepath, doc_name)
#     os.rename(doc_src, doc_dst)
# # for txt_name in file_name_list:
# #     name, ext = os.path.splitext(os.path.basename(txt_name))
# #     a = 'F:/dataset/ZZZX_ECG/AF/nor/'+name
# #     data = np.loadtxt(f'{a}.txt')
# #     b = 'F:/dataset/ZZZX_ECG/AF/npy/'+name
# #     np.save(f'{b}.npy', data)


# # 重采样并归一化
# import os
# import numpy as np
# from scipy import signal
#
#
# def mean_normalize(data):
#     mean_val = np.mean(data)
#     min_val = np.min(data)
#     max_val = np.max(data)
#     normalized_data = (data - mean_val) / (max_val - min_val)
#     return normalized_data
#
#
# txt_filepath = "F:/dataset/AF_2017/train/txt/"
# test_filepath = "H:/dataset/AF_2017/nor_2/"
# file_name_list = os.listdir(txt_filepath)
# for txt_name in file_name_list:
#     test_name = txt_name.split(".")[0] + ".txt"
#     in_file = open(txt_filepath + txt_name.split(".")[0] + ".txt", "r", newline="")
#     out_file = open(test_filepath + txt_name.split(".")[0] + ".txt", "w", newline="")
#     lines = in_file.readlines()  # 读取第一行
#     length = len(lines[0].strip().split())
#     # print(length)
#     new = length * (5 / 3)
#     # print(new)
#     new_num = int(new)
#     print(new_num)
#     data = []
#     for line in lines:
#         data = line.strip().split()  # 对每行的每个数据进行划分
#         data = np.array(data, dtype=np.float64) / 255.0
#         # data = np.array(data, dtype=np.float64)
#         data1 = signal.resample(data, new_num)
#         data2 = mean_normalize(data1)
#         print(' '.join(map(str, data2)), ' ', file=out_file)
#         # print(' '.join(map(str, data2)), ' ', file=out_file)
#         # print(' '.join(map(str, data2)), ' ', file=out_file)
#         # print(' '.join(map(str, data2)), ' ', file=out_file)
#         # print(' '.join(map(str, data2)), ' ', file=out_file)
#         # print(' '.join(map(str, data2)), ' ', file=out_file)
#         # print(' '.join(map(str, data2)), ' ', file=out_file)
#         # print(' '.join(map(str, data2)), ' ', file=out_file)
#         # print(' '.join(map(str, data2)), ' ', file=out_file)
#         # print(' '.join(map(str, data2)), ' ', file=out_file)
#         # print(' '.join(map(str, data2)), ' ', file=out_file)
#         # print(' '.join(map(str, data2)), ' ', file=out_file)
#     in_file.close()
#     out_file.close()
#     print("保存文件成功")


# # 提取PTB数据
# import numpy as np
# data = np.load('H:/dataset/archive/ecgeq-500hzsrfava.npy')
# for i in range(6428):
#     begin_num = i+1
#     name = str(begin_num)
#     if begin_num < 10:
#         name = 'A000' + name
#     elif 10 <= begin_num < 100:
#         name = 'A00' + name
#     elif 100 <= begin_num < 1000:
#         name = 'A0' + name
#     else:
#         name = 'A' + name
#     b = 'H:/dataset/archive/txt/'+name
#     np.savetxt(f'{b}.txt', data[i, :, :])


# torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False,
# threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
# torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)　
# from torch.optim import lr_scheduler

# scheduler = lr_scheduler.CosineAnnealingLR(optm, T_max=30, eta_min=0.001)

# # 挑选AF数据
# import numpy as np
# import os
# txt_filepath = "H:/dataset/PTB/npy/"
# file_name_list = os.listdir(txt_filepath)
# rename_format = '{}{}'
# label = np.load('H:/dataset/PTB/label.npy')
# i = 0
# for txt_name in file_name_list:
#     if label[i] == 1:
#         name, ext = os.path.splitext(os.path.basename(txt_name))
#         a = 'H:/dataset/PTB/npy/'+name
#         data = np.load(f'{a}.npy')
#         b = 'H:/dataset/PTB/N/'+name
#         np.save(f'{b}.npy', data)
#     i += 1


# # mat转npy
# import scipy.io
# import numpy as np
# mat = scipy.io.loadmat('H:/dataset/CSPC2018/TrainingSet/A0002.mat')
# data = mat['ECG']
# data = data['data']
# data = data[0][0]
# data = np.array(data, dtype=np.float64)
# np.savetxt('H:/dataset/CSPC2018/txt/A0001.txt', data)
# mat = scipy.io.loadmat('F:/dataset/AF_2017/training2017/A00001.mat')
# print(mat['val'])
# np.savetxt('H:/dataset/CSPC2018/txt/A0001.txt', mat['val'])


# import numpy as np
# from matplotlib import pyplot as plt
# from CPSC_config import Config
#
# class_num = 3
# config = Config()
# Conf_Mat_test = np.zeros((3, 3), dtype=int)
# Conf_Mat_test[0][0] = 154
# Conf_Mat_test[0][1] = 0
# Conf_Mat_test[0][2] = 106
# Conf_Mat_test[1][0] = 0
# Conf_Mat_test[1][1] = 216
# Conf_Mat_test[1][2] = 8
# Conf_Mat_test[2][0] = 74
# Conf_Mat_test[2][1] = 6
# Conf_Mat_test[2][2] = 491
# # 混淆矩阵图
# # 绘制混淆矩阵图
# fig, ax = plt.subplots()
# im = ax.imshow(Conf_Mat_test, cmap=plt.cm.Blues)
# # 添加标签和标题
# ax.set_xticks(np.arange(class_num))
# ax.set_yticks(np.arange(class_num))
# ax.set_xticklabels(config.CLASS_NAME, fontdict={'family': 'Times New Roman', 'size': 12})
# ax.set_yticklabels(config.CLASS_NAME, fontdict={'family': 'Times New Roman', 'size': 12})
# ax.set_xlabel('Predicted Label', fontdict={'family': 'Times New Roman', 'size': 12})
# ax.set_ylabel('True Label', fontdict={'family': 'Times New Roman', 'size': 12})
# ax.set_title('Confusion Matrix', fontdict={'family': 'Times New Roman', 'size': 14})
# # 添加注释
# thresh = Conf_Mat_test.max() / 2.
# for i in range(class_num):
#     for j in range(class_num):
#         ax.text(j, i, Conf_Mat_test[i, j],
#                 ha='center', va='center',
#                 color='white' if Conf_Mat_test[i, j] > thresh else 'black')
# # 显示图像
# fig.tight_layout()
# a = 'H:/dataset/try/eva_11/HX_1'
# fig.savefig(f'{a}.jpg', dpi=1200)


# import pandas as pd
# from sklearn.ensemble import BaggingClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.tree import DecisionTreeClassifier
# train = pd.read_excel('H:/dataset/PTB/age_sex.xlsx', index_col=0)
# test = pd.read_excel('H:/dataset/PTB/test.xlsx', index_col=0)
# X_train = train.drop(['result'], axis=1)
# y_train = train['result']
# X_test = test.drop(['result'], axis=1)
# y_test = test['result']
# # X_train = (X_train - np.min(X_train)) / (np.max(X_train) - np.min(X_train))
# bag_clf = BaggingClassifier(DecisionTreeClassifier(),
#                             n_estimators=500,
#                             max_samples=100,
#                             bootstrap=True,
#                             n_jobs=-1,
#                             random_state=42
#                             )
# bag_clf.fit(X_train, y_train)
# y_pred = bag_clf.predict(X_test)
# acc = accuracy_score(y_test, y_pred)
# print(acc)


# 批量mat提取数据
# import os
# import scipy.io
# import numpy as np
# txt_filepath = "F:/dataset/AF_2017/train/re_NO/"
# file_name_list = os.listdir(txt_filepath)
# rename_format = '{}{}'
# for txt_name in file_name_list:
#     name, ext = os.path.splitext(os.path.basename(txt_name))
#     a = 'F:/dataset/AF_2017/train/re_NO/'+name
#     data = scipy.io.loadmat(f'{a}.mat')
#     b = 'F:/dataset/AF_2017/train/1/'+name
#     np.savetxt(f'{b}.txt', data['ECG']['data'])


# import os
# import scipy.io
# import numpy as np
# txt_filepath = "H:/dataset/CSPC2018/TrainingSet/"
# file_name_list = os.listdir(txt_filepath)
# rename_format = '{}{}'
# for txt_name in file_name_list:
#     name, ext = os.path.splitext(os.path.basename(txt_name))
#     a = 'H:/dataset/CSPC2018/TrainingSet/'+name
#     data_1 = scipy.io.loadmat(f'{a}.mat')
#     data_2 = data_1['ECG']
#     data_3 = data_2['data']
#     b = 'H:/dataset/CSPC2018/txt/'+name
#     np.savetxt(f'{b}.txt', data_3[0][0])


# # 读取dat数据
# import numpy as np
# import os
# with open("F:/dataset/PTB-XL/ptb-xl-a/records100/00000/00001_lr.dat", "r") as f:
#     # data = f.read()
#     # print(data)
#     dat_content = f.readlines()
#     dat_content = [x.strip() for x in dat_content]
#     # np.save("F:/dataset/PTB-XL/ptb-xl-a/records100/00001_lr.npy")
#     # print(dat_content)


# 划分多个数据集
# import os
# import warnings
# import numpy as np
# from sklearn.model_selection import train_test_split
# from CPSC_config import Config
# import CPSC_utils as utils
#
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
# warnings.filterwarnings("ignore")
# config = Config()
#
# # 先平衡后合并
# # 第一个数据集
# records_name_1 = np.array(os.listdir(config.DATA_PATH_1))
# records_label_1 = np.load(config.REVISED_LABEL_1) - 1
# class_num = len(np.unique(records_label_1))
#
# # 划分训练，验证集 -----------------------------------------------------------------------------------------------
# train_records_1, val_records_1, train_labels_1, val_labels_1 = train_test_split(
#     records_name_1, records_label_1, test_size=0.2, random_state=config.RANDOM_STATE)
# num_categories_1 = []
# for i in range(class_num):
#     num_categories_1.append(len(train_labels_1[train_labels_1 == i]))
# print(num_categories_1)
#
# # 过采样使训练和验证集样本分布平衡 -------------------------------------------------------------------------------------
# # 1:1:1
# train_records_1, train_labels_1 = utils.oversample_balance(train_records_1, train_labels_1, config.RANDOM_STATE)
# num_categories_1 = []
# for i in range(class_num):
#     num_categories_1.append(len(train_labels_1[train_labels_1 == i]))
# print(num_categories_1)
#
# out_file_train = open('H:/dataset/AF_2017/' + "train_4.txt", "w", newline="")
# out_file_test = open('H:/dataset/AF_2017/' + "val_4.txt", "w", newline="")
# len_train = len(train_records_1)
# len_val = len(val_records_1)
# for i in range(len_train):
#     name_train, ext = os.path.splitext(os.path.basename(train_records_1[i]))
#     a = 'H:/dataset/AF_2017/train_3/' + name_train
#     data = np.load(f'{a}.npy')
#     if i+1 < 10:
#         b = 'H:/dataset/AF_2017/train_4/' + 'A000' + str(i+1)
#     elif 10 <= i+1 < 100:
#         b = 'H:/dataset/AF_2017/train_4/' + 'A00' + str(i+1)
#     elif 100 <= i+1 < 1000:
#         b = 'H:/dataset/AF_2017/train_4/' + 'A0' + str(i+1)
#     else:
#         b = 'H:/dataset/AF_2017/train_4/' + 'A' + str(i+1)
#     np.save(f'{b}.npy', data)
#     print(train_labels_1[i]+1, file=out_file_train)
# for j in range(len_val):
#     name_test, ext = os.path.splitext(os.path.basename(val_records_1[j]))
#     a = 'H:/dataset/AF_2017/train_3/' + name_test
#     data = np.load(f'{a}.npy')
#     if j+1 < 10:
#         b = 'H:/dataset/AF_2017/val_4/' + 'A000' + str(j+1)
#     elif 10 <= j+1 < 100:
#         b = 'H:/dataset/AF_2017/val_4/' + 'A00' + str(j+1)
#     elif 100 <= j+1 < 1000:
#         b = 'H:/dataset/AF_2017/val_4/' + 'A0' + str(j+1)
#     else:
#         b = 'H:/dataset/AF_2017/val_4/' + 'A' + str(j+1)
#     np.save(f'{b}.npy', data)
#     print(val_labels_1[j]+1, file=out_file_test)
# out_file_train.close()
# out_file_test.close()
#
# # 第二个数据集
# records_name_2 = np.array(os.listdir(config.DATA_PATH_2))
# records_label_2 = np.load(config.REVISED_LABEL_2) - 1
#
# # 划分训练，验证集 -----------------------------------------------------------------------------------------------
# train_records_2, val_records_2, train_labels_2, val_labels_2 = train_test_split(
#     records_name_2, records_label_2, test_size=0.2, random_state=config.RANDOM_STATE)
# num_categories_2 = []
# for i in range(class_num):
#     num_categories_2.append(len(train_labels_2[train_labels_2 == i]))
# print(num_categories_2)
#
# # 过采样使训练和验证集样本分布平衡 -------------------------------------------------------------------------------------
# # 1:1:1
# train_records_2, train_labels_2 = utils.oversample_balance(train_records_2, train_labels_2, config.RANDOM_STATE)
# num_categories_2 = []
# for i in range(class_num):
#     num_categories_2.append(len(train_labels_2[train_labels_2 == i]))
# print(num_categories_2)
#
# out_file_train = open('H:/dataset/CSPC2018/' + "train_3.txt", "w", newline="")
# out_file_test = open('H:/dataset/CSPC2018/' + "val_3.txt", "w", newline="")
# len_train = len(train_records_2)
# len_val = len(val_records_2)
# for i in range(len_train):
#     name_train, ext = os.path.splitext(os.path.basename(train_records_2[i]))
#     a = 'H:/dataset/CSPC2018/train_2/' + name_train
#     data = np.load(f'{a}.npy')
#     if i+1 < 10:
#         b = 'H:/dataset/CSPC2018/train_3/' + 'AA000' + str(i+1)
#     elif 10 <= i+1 < 100:
#         b = 'H:/dataset/CSPC2018/train_3/' + 'AA00' + str(i+1)
#     elif 100 <= i+1 < 1000:
#         b = 'H:/dataset/CSPC2018/train_3/' + 'AA0' + str(i+1)
#     else:
#         b = 'H:/dataset/CSPC2018/train_3/' + 'AA' + str(i+1)
#     np.save(f'{b}.npy', data)
#     print(train_labels_2[i]+1, file=out_file_train)
# for j in range(len_val):
#     name_test, ext = os.path.splitext(os.path.basename(val_records_2[j]))
#     a = 'H:/dataset/CSPC2018/train_2/' + name_test
#     data = np.load(f'{a}.npy')
#     if j+1 < 10:
#         b = 'H:/dataset/CSPC2018/val_3/' + 'AA000' + str(j+1)
#     elif 10 <= j+1 < 100:
#         b = 'H:/dataset/CSPC2018/val_3/' + 'AA00' + str(j+1)
#     elif 100 <= j+1 < 1000:
#         b = 'H:/dataset/CSPC2018/val_3/' + 'AA0' + str(j+1)
#     else:
#         b = 'H:/dataset/CSPC2018/val_3/' + 'AA' + str(j+1)
#     np.save(f'{b}.npy', data)
#     print(val_labels_2[j]+1, file=out_file_test)
# out_file_train.close()
# out_file_test.close()
#
# # 第三个数据集
# records_name_3 = np.array(os.listdir(config.DATA_PATH_3))
# records_label_3 = np.load(config.REVISED_LABEL_3) - 1
#
# # 划分训练，验证集 -----------------------------------------------------------------------------------------------
# train_records_3, val_records_3, train_labels_3, val_labels_3 = train_test_split(
#     records_name_3, records_label_3, test_size=0.2, random_state=config.RANDOM_STATE)
# num_categories_3 = []
# for i in range(class_num):
#     num_categories_3.append(len(train_labels_3[train_labels_3 == i]))
# print(num_categories_3)
#
# # 过采样使训练和验证集样本分布平衡 -------------------------------------------------------------------------------------
# # 1:1:1
# train_records_3, train_labels_3 = utils.oversample_balance(train_records_3, train_labels_3, config.RANDOM_STATE)
# num_categories_3 = []
# for i in range(class_num):
#     num_categories_3.append(len(train_labels_3[train_labels_3 == i]))
# print(num_categories_3)
#
# out_file_train = open('H:/dataset/YY2023/' + "train_3.txt", "w", newline="")
# out_file_test = open('H:/dataset/YY2023/' + "val_3.txt", "w", newline="")
# len_train = len(train_records_3)
# len_val = len(val_records_3)
# for i in range(len_train):
#     name_train, ext = os.path.splitext(os.path.basename(train_records_3[i]))
#     a = 'H:/dataset/YY2023/train_2/' + name_train
#     data = np.load(f'{a}.npy')
#     if i+1 < 10:
#         b = 'H:/dataset/YY2023/train_3/' + 'AAA000' + str(i+1)
#     elif 10 <= i+1 < 100:
#         b = 'H:/dataset/YY2023/train_3/' + 'AAA00' + str(i+1)
#     elif 100 <= i+1 < 1000:
#         b = 'H:/dataset/YY2023/train_3/' + 'AAA0' + str(i+1)
#     else:
#         b = 'H:/dataset/YY2023/train_3/' + 'AAA' + str(i+1)
#     np.save(f'{b}.npy', data)
#     print(train_labels_3[i]+1, file=out_file_train)
# for j in range(len_val):
#     name_test, ext = os.path.splitext(os.path.basename(val_records_3[j]))
#     a = 'H:/dataset/YY2023/train_2/' + name_test
#     data = np.load(f'{a}.npy')
#     if j+1 < 10:
#         b = 'H:/dataset/YY2023/val_3/' + 'AAA000' + str(j+1)
#     elif 10 <= j+1 < 100:
#         b = 'H:/dataset/YY2023/val_3/' + 'AAA00' + str(j+1)
#     elif 100 <= j+1 < 1000:
#         b = 'H:/dataset/YY2023/val_3/' + 'AAA0' + str(j+1)
#     else:
#         b = 'H:/dataset/YY2023/val_3/' + 'AAA' + str(j+1)
#     np.save(f'{b}.npy', data)
#     print(val_labels_3[j]+1, file=out_file_test)
# out_file_train.close()
# out_file_test.close()
#
# # 第四个数据集
# records_name_4 = np.array(os.listdir(config.DATA_PATH_4))
# records_label_4 = np.load(config.REVISED_LABEL_4) - 1
#
# # 划分训练，验证集 -----------------------------------------------------------------------------------------------
# train_records_4, val_records_4, train_labels_4, val_labels_4 = train_test_split(
#     records_name_4, records_label_4, test_size=0.2, random_state=config.RANDOM_STATE)
# num_categories_4 = []
# for i in range(class_num):
#     num_categories_4.append(len(train_labels_4[train_labels_4 == i]))
# print(num_categories_4)
#
# # 过采样使训练和验证集样本分布平衡 -------------------------------------------------------------------------------------
# # 1:1:1
# train_records_4, train_labels_4 = utils.oversample_balance(train_records_4, train_labels_4, config.RANDOM_STATE)
# num_categories_4 = []
# for i in range(class_num):
#     num_categories_4.append(len(train_labels_4[train_labels_4 == i]))
# print(num_categories_4)
#
# out_file_train = open('H:/dataset/PTB/' + "train_3.txt", "w", newline="")
# out_file_test = open('H:/dataset/PTB/' + "val_3.txt", "w", newline="")
# len_train = len(train_records_4)
# len_val = len(val_records_4)
# for i in range(len_train):
#     name_train, ext = os.path.splitext(os.path.basename(train_records_4[i]))
#     a = 'H:/dataset/PTB/train_2/' + name_train
#     data = np.load(f'{a}.npy')
#     if i+1 < 10:
#         b = 'H:/dataset/PTB/train_3/' + 'AAAA000' + str(i+1)
#     elif 10 <= i+1 < 100:
#         b = 'H:/dataset/PTB/train_3/' + 'AAAA00' + str(i+1)
#     elif 100 <= i+1 < 1000:
#         b = 'H:/dataset/PTB/train_3/' + 'AAAA0' + str(i+1)
#     else:
#         b = 'H:/dataset/PTB/train_3/' + 'AAAA' + str(i+1)
#     np.save(f'{b}.npy', data)
#     print(train_labels_4[i]+1, file=out_file_train)
# for j in range(len_val):
#     name_test, ext = os.path.splitext(os.path.basename(val_records_4[j]))
#     a = 'H:/dataset/PTB/train_2/' + name_test
#     data = np.load(f'{a}.npy')
#     if j+1 < 10:
#         b = 'H:/dataset/PTB/val_3/' + 'AAAA000' + str(j+1)
#     elif 10 <= j+1 < 100:
#         b = 'H:/dataset/PTB/val_3/' + 'AAAA00' + str(j+1)
#     elif 100 <= j+1 < 1000:
#         b = 'H:/dataset/PTB/val_3/' + 'AAAA0' + str(j+1)
#     else:
#         b = 'H:/dataset/PTB/val_3/' + 'AAAA' + str(j+1)
#     np.save(f'{b}.npy', data)
#     print(val_labels_4[j]+1, file=out_file_test)
# out_file_train.close()
# out_file_test.close()
