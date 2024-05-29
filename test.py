# 测试

import os
import warnings
import numpy as np
import my_utils
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix, accuracy_score
from my_utils import Config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
warnings.filterwarnings("ignore")
config = Config()

# 测试集
test_records = np.array(os.listdir(config.DATA))
test_labels = np.load(config.LABEL) - 1
class_num = len(np.unique(test_labels))

# 测试（单导联）
LEAD = 1
TARGET_LEAD = 1
# test_x = my_utils.Fetch_Pats_Lbs_sLead(test_records, Path=config.DATA,
#                                     seg_num=config.SEG_NUM,
#                                     seg_length=config.SEG_LENGTH)
# test_y = to_categorical(test_labels, num_classes=class_num)
test_x = my_utils.Fetch_Pats_Lbs_sLead(test_records, Path=config.DATA,
                                       target_lead=LEAD, seg_num=config.SEG_NUM,
                                       seg_length=config.SEG_LENGTH)
test_y = to_categorical(test_labels, num_classes=class_num)
for j in range(test_x.shape[0]):
    test_x[j, :, :] = scale(test_x[j, :, :], axis=0)
# i = 5
model_name = 'net_lead_' + str(TARGET_LEAD) + '.hdf5'
# model_name = 'net_lead_' + str(TARGET_LEAD) + '_' + str(i) + '.hdf5'
model = load_model(config.MODEL_PATH + model_name)

print('\nResult for Lead ' + str(LEAD) + '-----------------------------\n')
pred_vt = model.predict(test_x)
# print(pred_vt)
pred_v = np.argmax(pred_vt, axis=1)
print(pred_v)
true_v = np.argmax(test_y, axis=1)
# print(test_y)
print(true_v)

# 绘制评价图像
PATH = 'H:/dataset/try/test/test_87/'
# t = str(TARGET_LEAD) + '_' + str(i)
# utils.evaluation(pred_vt, test_y, true_v, pred_v, class_num, t, PATH)
my_utils.evaluation(pred_vt, test_y, true_v, pred_v, class_num, LEAD, PATH)
# del test_x, test_y

# 评估模型在测试集上的性能
Conf_Mat_test = confusion_matrix(true_v, pred_v)  # 真实值和预测值的混淆矩阵
print(Conf_Mat_test)
F1s_test = []
for j in range(class_num):
    f1t = 2 * Conf_Mat_test[j][j] / (np.sum(Conf_Mat_test[j, :]) + np.sum(Conf_Mat_test[:, j]))
    print('| F1-' + config.CLASS_NAME[j] + ':' + str(f1t) + ' |')
    F1s_test.append(f1t)
print('F1-mean: ' + str(np.mean(F1s_test)))
for j in range(class_num):
    acc = Conf_Mat_test[j][j] / np.sum(Conf_Mat_test[:, j])
    print('| P-' + config.CLASS_NAME[j] + ':' + str(acc) + ' |')
print('ACC: ', accuracy_score(true_v, pred_v))
