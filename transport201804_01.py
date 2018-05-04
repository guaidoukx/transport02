# **coding:UTF-8**
import pandas as pd
import math

example_tr = pd.DataFrame(pd.read_csv('IO/data201804.csv',
                                   usecols=['A', 'B', 'E1', 'E2', 'E3', 'F2', 'F3', 'F4', 'G11',
                                             'G12', 'G13', 'G14', 'G21', 'G22', 'G23', 'G24', 'G25',
                                             'G26', 'G31', 'G32', 'G33',  'G35', 'J', 'C', 'D', 'F1', 'H', 'I']))
example_te = pd.DataFrame(pd.read_csv('IO/test201804.csv',
                                   usecols=['A', 'B', 'E1', 'E2', 'E3', 'F2', 'F3', 'F4', 'G11',
                                             'G12', 'G13', 'G14', 'G21', 'G22', 'G23', 'G24', 'G25',
                                             'G26', 'G31', 'G32', 'G33',  'G35', 'J', 'C', 'D', 'F1', 'H', 'I']))

class Data(object):
    def __init__(self,data):
        self.data = data

    def print_info(self):
        print(self.data.info())

    def get_columns(self):
        return self.data.columns

    def cols_in_type(self, type):
        L = []
        if type == 'num':
            for i in self.get_columns():
                if self.data[i].dtype == 'float64' or self.data[i].dtype == 'int64':
                    L.append(i)
            return L
        elif type == 'str':
            for i in self.get_columns():
                if self.data[i].dtype == 'object':
                    L.append(i)
            return L
        else:
            print("Error Parameter. It must be 'str' or 'num'.")

    def count(self,object_name):
        counts = self.data[object_name].value_counts()
        quants = counts.values
        return sum(quants)

    def print_count(self, object_name):
        counts = self.data[object_name].value_counts()
        quants = counts.values
        print('----' + object_name + '-----')
        print('sum:   %d' % sum(quants))
        print(counts)
        print('\r')


    def average(self, object_name):
        sum = 0
        cou = 0
        for i in self.data[object_name]:
            if not(i != i):
                sum = sum + i
                cou = cou + 1
        # print(object_name + '的平均值为：%f' % (sum / cou)+'\r')
        return (sum / cou)

    def normalize(self, object_name):
        dif = max(self.data[object_name]) - min(self.data[object_name])
        aver = self.average(object_name)
        if dif == 0:
            self.data[object_name] = self.data[object_name].apply(lambda x: 0)
        else:
            self.data[object_name] = self.data[object_name].apply(lambda x: (x - aver) / dif)

    def smooth(self, object_name):
        self.data[object_name] = self.data[object_name].apply(lambda x: math.log(x))



trdata = Data(example_tr)
trdata.items = len(trdata.data)
trdata.num_cols = trdata.cols_in_type('num')
trdata.str_cols = trdata.cols_in_type('str')
trdata.cols = trdata.num_cols + trdata.str_cols

tedata = Data(example_te)
tedata.items = len(tedata.data)

# 将所有数值型列空值填成改列的平均值，并归一化，还没有做平滑=================
trdata.num_cols.remove('A')
for i in trdata.num_cols:
    if trdata.count(i) != trdata.items:
        trdata.data[i] = trdata.data[i].fillna(trdata.average(i))
    # tedata.print_count(i)
    if tedata.count(i) != tedata.items:
        tedata.data[i] = tedata.data[i].fillna(trdata.average(i))
    # tedata.print_count(i)
    # trdata.normalize(i)
    # tedata.normalize(i)
    # tedata.print_count(i)

# 非数值型列没有空值，只需要转化为one-hot形式即可=========================
trdata.combine = pd.concat([trdata.data, tedata.data])
trdata.combine = pd.get_dummies(trdata.combine)
trdata.after = trdata.combine[0:trdata.items]
tedata.after = trdata.combine[trdata.items:]

pd.DataFrame.to_csv(trdata.after, 'IO/train_after.csv', index=None)
pd.DataFrame.to_csv(tedata.after, 'IO/test_after.csv', index=None)