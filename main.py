import os
import warnings
warnings.filterwarnings("ignore")
import sklearn.neighbors._base
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
import math
import pandas as pd
import numpy as np
import sklearn
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score, matthews_corrcoef
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN, KMeansSMOTE, SMOTENC, SVMSMOTE
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours\
    , RepeatedEditedNearestNeighbours, AllKNN, CondensedNearestNeighbour, OneSidedSelection, NeighbourhoodCleaningRule\
    , InstanceHardnessThreshold
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import EasyEnsemble, BalanceCascade
# from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import datetime
import pandas as pd

MAP_PATH = 'SCA warining data/important feature id mapping.xlsx'

def get_feature_ids():
    '''
    获取表格中的程序特征id
    :return:
    '''
    df = pd.read_excel(MAP_PATH)
    notes = df['Note']
    program_ids = df['id in the program']
    features_ids = []
    for index, note in enumerate(notes):
        if note == 1.0:
            program_id = program_ids[index]
            features_ids.append(program_id)
    return features_ids

def string2int(string_ls):
    '''
    将特征中是string的转换为int
    :param string_ls:
    :return:
    '''
    string_set = set(string_ls)
    category = range(len(string_set))
    map_dict = dict(zip(string_set, category))
    int_ls = []
    for i in string_ls:
        int_ls.append(map_dict[i])
    return int_ls

def get_data(features_ids, data_path = 'SCA warining data/ant/test_set/totalFeatures5.csv'):
    '''
    获取features_ids中包含的feature；并且将类别转换为1,0,open=1,close=0；并将string类型的转换为int
    :param features_ids: 需要的特征id
    :param data_path: 原始数据path
    :return: feature_data x数据；y 标签
    '''
    df = pd.read_csv(data_path).sample(frac=1)
    labels = df['category'].values.tolist()
    y = []
    for label in labels:
        if label == 'open':
            y.append(1)
        else:
            y.append(0)
    feature_data = []
    df_cl_name = df.columns.values.tolist()
    for cl in df_cl_name:
        for id in features_ids:
            if str(id) in cl:
                cl_list = df[cl].values.tolist()
                if isinstance(cl_list[0], str):
                    transform_cl_list = string2int(cl_list)
                    feature_data.append(transform_cl_list)
                else:
                    feature_data.append(cl_list)

    # print(feature_data)
    # print(y)
    feature_data = np.array(feature_data).transpose()
    y = np.array(y)
    return feature_data, y


def train(x, y, classifier):
    # train_data, test_data, train_label, test_label = sklearn.model_selection.train_test_split(x, y, random_state=1, train_size=0.9, test_size=0.1)
    # classifier = svm.SVC(kernel='rbf') #'poly', 'rbf', 'linear'
    # classifier.fit(train_data, train_label.ravel()) # ravel函数在降维时默认是行序优先
    # predict = classifier.predict(test_data)
    predict = cross_val_predict(classifier, x, y, cv=10)
    tn, fp, fn, tp = confusion_matrix(y, predict).ravel()
    # print(test_label)
    # print(predict)
    acc, pre, recall, f1, auc, mcc = accuracy_score(y, predict), precision_score(y, predict), \
                           recall_score(y, predict), f1_score(y, predict), roc_auc_score(y, predict), \
                           matthews_corrcoef(y, predict)
    g_measure = math.sqrt(pre*recall)
    print('TP:'+str(tp), 'FP:'+str(fp), 'FN:'+str(fn), 'TN:'+str(tn))
    print('Acc:'+str(acc), 'Pre:'+str(pre), 'Recall:'+str(recall), 'F1:'+str(f1), 'AUC:'+str(auc), 'Mcc:'+str(mcc),
          'G measure:'+str(g_measure))
    return tp, fp, fn, tn, float(format(acc, '.3f')), float(format(pre, '.3f')), float(format(recall, '.3f')), \
           float(format(f1, '.3f')), float(format(auc, '.3f')), float(format(mcc, '.3f')), float(format(g_measure, '.3f'))
    # # 4.计算svc分类器的准确率
    # print("训练集：", classifier.score(train_data, train_label))
    # print("测试集：", classifier.score(test_data, test_label))

def over_sample(x,y, classifier):
    '''
    过采样学习
    '''
    print('过采样')
    for os_method in over_sampler_methods:
        starttime = datetime.datetime.now()
        x_resampled, y_resampled = os_method.fit_resample(x, y)
        endtime = datetime.datetime.now()
        print(str(os_method)+' Time', (endtime - starttime).seconds)
        train(x_resampled, y_resampled, classifier)

def under_sampling_sample(x, y, classifier):
    '''
    欠采样学习
    '''
    print('欠采样')
    for os_method in under_sampler_methods:
        starttime = datetime.datetime.now()
        x_resampled, y_resampled = os_method.fit_resample(x, y)
        endtime = datetime.datetime.now()
        print(str(os_method) + ' Time', (endtime - starttime).seconds)
        train(x_resampled, y_resampled, classifier)


def combine_sampling_sample(x, y, classifier):
    '''
    组合采样学习
    '''
    print('组合采样')
    combine_sampler_methods = [SMOTEENN(), SMOTETomek()]
    for os_method in combine_sampler_methods:
        starttime = datetime.datetime.now()
        x_resampled, y_resampled = os_method.fit_resample(x, y)
        endtime = datetime.datetime.now()
        print(str(os_method) + ' Time', (endtime - starttime).seconds)
        train(x_resampled, y_resampled, classifier)

def transform_numpy2list(x, y):
    '''
    将二维numpy转换为二维list
    :param x:
    :param y:
    :return:
    '''
    x_ls, y_ls = [], []
    for index,i in enumerate(x):
        tag_ls = []
        # tag_ls.append(str(index))
        for j in i:
            if not isinstance(str(j),str):
                print(j, type(str(j)))
            tag_ls.append(str(j))
        x_ls.append(tag_ls)
    for t in y:
        y_ls.append(str(t))
    return x_ls, y_ls#np.array(x), np.array(y)


def ensemble_sampling_sample(x, y, classifier):
    '''
    集成采样学习
    '''
    print('集成采样') #EasyEnsemble(),
    starttime = datetime.datetime.now()
    x_resampled, y_resampled = EasyEnsemble().fit_resample(x, y)
    endtime = datetime.datetime.now()
    print(str(EasyEnsemble) + ' Time', (endtime - starttime).seconds)
    train(x_resampled.reshape(-1, 93), y_resampled.reshape(-1), classifier)
    # ensemble_sampler_methods = [BalanceCascade()]
    # starttime = datetime.datetime.now()
    # new_x, new_y = transform_numpy2list(x, y)
    # # new_y = list(y)
    # x_resampled, y_resampled = BalanceCascade().fit_resample(new_x, new_y)
    # endtime = datetime.datetime.now()
    # print(str(EasyEnsemble) + ' Time', (endtime - starttime).seconds)
    # train(x_resampled.reshape(-1, 93), y_resampled.reshape(-1))



if __name__ == '__main__':
    writer = pd.ExcelWriter("unbalance_learn.xlsx")
    project_data = {}
    feature_ids = get_feature_ids()
    dir_ls = os.listdir('SCA warining data')
    for dir_name in dir_ls[4:]:
        if '.' in dir_name: continue
        # dir_name = 'commons'
        x, y = get_data(feature_ids, 'SCA warining data/'+dir_name+'/'+'test_set/totalFeatures5.csv')
        classifier_ls = [svm.SVC(kernel='rbf'), RandomForestClassifier(), GaussianNB(), KNeighborsClassifier()
            , DecisionTreeClassifier(), LogisticRegression(solver='liblinear'), AdaBoostClassifier()]#[]
        over_sampler_methods = [RandomOverSampler(), SMOTE(), BorderlineSMOTE(), ADASYN(),
                                SMOTENC(categorical_features=[0, 1]), SVMSMOTE()]
        under_sampler_methods = [ClusterCentroids(), RandomUnderSampler()
            , NearMiss(), TomekLinks(), EditedNearestNeighbours()
            , RepeatedEditedNearestNeighbours(), AllKNN(), CondensedNearestNeighbour(), OneSidedSelection()
            , NeighbourhoodCleaningRule(), InstanceHardnessThreshold()]
        combine_sampler_methods = [SMOTEENN(), SMOTETomek()]
        model_ls, unbalance_category_ls, unbalance_method_ls, balance_data_time_ls, tp_ls_ls, fp_ls, fn_ls, \
        tn_ls, acc_ls, pre_ls, recall_ls, f1_ls, auc_ls, mcc_ls, g_measure_ls = [[] for x in range(15)]
        # model, unbalance_category, unbalance_method, balance_data_time, tp, fp, fn, tn, acc, pre, recall, f1, auc, mcc, g_measure = [[] for x in range(15)]
        for index, classifier in enumerate(classifier_ls):
            # print('-------', index)
            tp, fp, fn, tn, acc, pre, recall, f1, auc, mcc, g_measure = train(x, y, classifier) #无不平衡学习
            model_ls.append(str(classifier).split('(')[0]), unbalance_category_ls.append('None'), unbalance_method_ls.append('None'), balance_data_time_ls.append('None')
            tp_ls_ls.append(tp), fp_ls.append(fp), fn_ls.append(fn), tn_ls.append(tn), acc_ls.append(acc)
            pre_ls.append(pre), recall_ls.append(recall), f1_ls.append(f1), auc_ls.append(auc), mcc_ls.append(mcc), g_measure_ls.append(g_measure)
            for over_method in over_sampler_methods:
                starttime = datetime.datetime.now()
                x_resampled, y_resampled = over_method.fit_resample(x, y)
                endtime = datetime.datetime.now()
                balance_data_time = (endtime - starttime).seconds
                # print(str(os_method) + ' Time', (endtime - starttime).seconds)
                tp, fp, fn, tn, acc, pre, recall, f1, auc, mcc, g_measure = train(x_resampled, y_resampled, classifier)
                model_ls.append(str(classifier).split('(')[0]), unbalance_category_ls.append(
                    'over_sampler'), unbalance_method_ls.append(str(over_method).split('(')[0]), balance_data_time_ls.append(balance_data_time)
                tp_ls_ls.append(tp), fp_ls.append(fp), fn_ls.append(fn), tn_ls.append(tn), acc_ls.append(acc)
                pre_ls.append(pre), recall_ls.append(recall), f1_ls.append(f1), auc_ls.append(auc), mcc_ls.append(
                    mcc), g_measure_ls.append(g_measure)

            for under_method in under_sampler_methods:
                starttime = datetime.datetime.now()
                x_resampled, y_resampled = under_method.fit_resample(x, y)
                endtime = datetime.datetime.now()
                balance_data_time = (endtime - starttime).seconds
                # print(str(os_method) + ' Time', (endtime - starttime).seconds)
                tp, fp, fn, tn, acc, pre, recall, f1, auc, mcc, g_measure = train(x_resampled, y_resampled, classifier)
                model_ls.append(str(classifier).split('(')[0]), unbalance_category_ls.append(
                    'under_sampler'), unbalance_method_ls.append(
                    str(under_method).split('(')[0]), balance_data_time_ls.append(balance_data_time)
                tp_ls_ls.append(tp), fp_ls.append(fp), fn_ls.append(fn), tn_ls.append(tn), acc_ls.append(acc)
                pre_ls.append(pre), recall_ls.append(recall), f1_ls.append(f1), auc_ls.append(auc), mcc_ls.append(
                    mcc), g_measure_ls.append(g_measure)

            for combine_method in combine_sampler_methods:
                starttime = datetime.datetime.now()
                x_resampled, y_resampled = combine_method.fit_resample(x, y)
                endtime = datetime.datetime.now()
                balance_data_time = (endtime - starttime).seconds
                # print(str(os_method) + ' Time', (endtime - starttime).seconds)
                tp, fp, fn, tn, acc, pre, recall, f1, auc, mcc, g_measure = train(x_resampled, y_resampled, classifier)
                model_ls.append(str(classifier).split('(')[0]), unbalance_category_ls.append(
                    'combine_sampler'), unbalance_method_ls.append(
                    str(combine_method).split('(')[0]), balance_data_time_ls.append(balance_data_time)
                tp_ls_ls.append(tp), fp_ls.append(fp), fn_ls.append(fn), tn_ls.append(tn), acc_ls.append(acc)
                pre_ls.append(pre), recall_ls.append(recall), f1_ls.append(f1), auc_ls.append(auc), mcc_ls.append(
                    mcc), g_measure_ls.append(g_measure)
            starttime = datetime.datetime.now()
            x_resampled, y_resampled = EasyEnsemble().fit_resample(x, y)
            endtime = datetime.datetime.now()
            balance_data_time = (endtime - starttime).seconds
            # print(str(EasyEnsemble) + ' Time', (endtime - starttime).seconds)
            print(dir_name)
            tp, fp, fn, tn, acc, pre, recall, f1, auc, mcc, g_measure = train(x_resampled.reshape(len(y_resampled.reshape(-1)), -1), y_resampled.reshape(-1), classifier)
            model_ls.append(str(classifier).split('(')[0]), unbalance_category_ls.append(
                'ensemble_sampler'), unbalance_method_ls.append(
                'EasyEnsemble'), balance_data_time_ls.append(balance_data_time)
            tp_ls_ls.append(tp), fp_ls.append(fp), fn_ls.append(fn), tn_ls.append(tn), acc_ls.append(acc)
            pre_ls.append(pre), recall_ls.append(recall), f1_ls.append(f1), auc_ls.append(auc), mcc_ls.append(
                mcc), g_measure_ls.append(g_measure)
        project_data['model'] = model_ls; project_data['unbalance_category'] = unbalance_category_ls;
        project_data['unbalance_method'] = unbalance_method_ls; project_data['balance_data_time'] = balance_data_time_ls
        project_data['TP'] = tp_ls_ls; project_data['FP'] = fp_ls; project_data['FN'] = fn_ls; project_data['TN'] = tn_ls
        project_data['ACC'] = acc_ls; project_data['PRE'] = pre_ls; project_data['RECALL'] = recall_ls
        project_data['F1'] = f1_ls; project_data['AUC'] = auc_ls; project_data['MCC'] = mcc_ls
        project_data['G_measure'] = g_measure_ls
        data = pd.DataFrame(project_data)
        data.to_excel(writer, sheet_name = dir_name)
        data.to_csv(dir_name+'.csv')
    writer.save()
    writer.close()
        # over_sample(x, y, classifier)
        # under_sampling_sample(x, y, classifier)
        # combine_sampling_sample(x, y, classifier)
        # ensemble_sampling_sample(x, y, classifier)
