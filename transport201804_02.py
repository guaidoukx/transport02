# **coding:UTF-8**
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_validation import KFold, train_test_split
from keras.layers.core import Dense
from keras.models import Sequential
import xgboost as xgb

example_trn = pd.read_csv('IO/train_after_n.csv')
example_ten = pd.read_csv('IO/test_after_n.csv')
example_tr = pd.read_csv('IO/train_after.csv')
example_te = pd.read_csv('IO/test_after.csv')


def MAD(target, predictions):
    absolute_deviation = np.abs(target - predictions)
    return np.mean(absolute_deviation)


class ReadData(object):
    def __init__(self, data, col_Y):
        self.data = data
        self.col_Y = col_Y
    
    def split_to_kfold(self, overall, split):
        pdY = self.data[self.col_Y]
        pdX = self.data.drop([self.col_Y], axis=1)
        index = np.arange(0, len(pdY))
        np.random.shuffle(index)
        Y = pdY.values
        X = pdX.values
        kf = KFold(overall, n_folds=split, shuffle=True)
        L = []
        for tr_index, te_index in kf:
            tr_X = X[tr_index]
            tr_Y = Y[tr_index]
            te_X = X[te_index]
            te_Y = Y[te_index]
            L.append([tr_X, tr_Y, te_X, te_Y])
        self.L, self.kf = L, kf
    
    def Train(self, model_name, tr_X, tr_Y):
        if model_name == 'SVR':
            models = SVR(kernel='poly', degree=4, gamma=1.8)
            models.fit(tr_X, tr_Y)
            self.model = models
        elif model_name == 'NN':
            models = Sequential()
            models.add(Dense(input_dim=62, units=500, activation='relu'))
            models.add(Dense(units=1000, activation='relu'))
            models.add(Dense(units=1000, activation='relu'))
            models.add(Dense(units=500, activation='relu'))
            models.add(Dense(units=200, activation='relu'))
            models.add(Dense(units=100, activation='relu'))
            models.add(Dense(units=1))
            models.compile(loss='mse', optimizer='Adam')
            models.fit(tr_X, tr_Y, batch_size=100, epochs=70)
            self.model = models
        elif model_name == 'XGBOOST':
            train_x, val_x = train_test_split(tr_X, test_size=0.2, random_state=20)
            train_y, val_y = train_test_split(tr_Y, test_size=0.2, random_state=20)
            xgb_train = xgb.DMatrix(train_x, train_y)
            xgb_val = xgb.DMatrix(val_x, val_y)
            self.parameters = {
                'booster' : 'gbtree',
                'objective' : 'reg:linear',
                'eval_metric': 'rmse',
                'gamma': 0.1,
                'max_depth': 8,
                'subsample':0.9,
                'colsample_bytree': 0.9,
                'eta': 0.06
            }
            self.num_rounds = 150
            self.watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
            self.model = xgb.train(self.parameters, xgb_train, self.num_rounds, self.watchlist, early_stopping_rounds=100)
        else:
            print("Error Parameter! It must be 'SVR', 'NN' or 'XGBOOST'. " )
    
    def evaluation(self, Y, pred, way):
        if way =='RMSE':
            print("Root Mean Squared Error: %.4f" % np.sqrt(mean_squared_error(Y, pred)))
        elif way == 'R2':
            print('R2 score: %.4f' % r2_score(Y,pred))
        elif way == 'MAD':
            print("Mean Absolute Deviation: %.4f" % MAD(Y, pred) + "\r" )
        else:
            print("Error Parameter! It must be 'RMSE','R2' or 'MAD'.")
    
    def printResult(self, model_name,tr_X, tr_Y, te_X, te_Y):
        if model_name == 'NN':
            result = self.model.evaluate(te_X, te_Y)
            # print('\nNeural Network Training Finish & result')
            # print('Test result loss:', result)
        tr_pred = self.model.predict(tr_X)
        te_pred = self.model.predict(te_X)
        
        print('---' + model_name + ' Train---')
        self.evaluation(tr_Y, tr_pred, 'RMSE')
        self.evaluation(tr_Y, tr_pred, 'R2')
        self.evaluation(tr_Y, tr_pred, 'MAD')
        print('---' + model_name + ' Test---')
        self.evaluation(te_Y, te_pred, 'RMSE')
        self.evaluation(te_Y, te_pred, 'R2')
        self.evaluation(te_Y, te_pred, 'MAD')
        self.te_y = te_Y.tolist()
        self.te_pred = te_pred.flatten().tolist()
    
    def train_test_print(self,  model_name):
        self.D = {-1:([],[])}
        for t, pred in enumerate(self.kf):
            tr_X, tr_Y, te_X, te_Y = self.L[t][0], self.L[t][1], self.L[t][2], self.L[t][3]
            print('\n----%d iteration of ' % (t + 1) + model_name + ' ----')
            self.printResult(model_name, tr_X, tr_Y, te_X, te_Y)
            self.D[t] = (self.D[t-1][0]+self.te_y, self.D[t-1][1]+self.te_pred)
            # print(self.D[t])
        print('\n--- '+ model_name+' final result---')
        self.evaluation(self.D[4][0], self.D[4][1],'RMSE')
        self.evaluation(self.D[4][0], self.D[4][1],'R2')
        # self.evaluation(np.ndarray(self.D[4][0]), np.ndarray(self.D[4][1]),'MAD')

    def XG(self):
        self.D = {-1: ([], [])}
        for t, pred in enumerate(self.kf):
            tr_X, tr_Y, te_X, te_Y = self.L[t][0], self.L[t][1], self.L[t][2], self.L[t][3]
            self.te_pred = self.model.predict(xgb.DMatrix(te_X)).tolist()
            # print(type(self.te_pred))
            # print(type(te_Y))
            # print(len(self.te_pred))
            # print(len(te_Y))
            # print(self.te_pred)
            # print(te_Y)
            # print(self.te_pred.shape)
            # print(te_Y.shape)
            print('\n-----%d iteration xgboost Training Finish & result-----' % (t + 1))
            print("Root Mean Squared Error: %.4f" % np.sqrt(mean_squared_error(te_Y, self.te_pred)))
            print('R2 score: %.4f' % r2_score(te_Y, self.te_pred))
            self.D[t] = (self.D[t-1][0] + te_Y.tolist(), self.D[t-1][1] + self.te_pred)
        
        print("\n-----====final xgboost result===-----")
        # print("final test Root Mean Squared Error: %.4f" % self.evaluation(np.array(self.D[4][0]), np.array(self.D[4][1]),'RMSE'))
        # print('final test R2 score: %.4f' % self.evaluation(np.array(self.D[4][0]), np.array(self.D[4][1]),'R2'))
        # print('final test R2 score: %.4f' % self.evaluation(pd.Series(self.D[4][0]), pd.Series(self.D[4][1]),'R2'))



trdata = ReadData(example_trn, 'A')
trdata.overall = len(trdata.data)
trdata.split_to_kfold(trdata.overall, 5)
trdata.Train('NN', trdata.L[0][0],trdata.L[0][1])
trdata.train_test_print('NN')
print(trdata.model.predict(example_ten.drop(['A'],axis=1)))
trdata.Train('SVR', trdata.L[0][0],trdata.L[0][1])
trdata.train_test_print('SVR')
print(trdata.model.predict(example_ten.drop(['A'],axis=1)))


# xgdata = ReadData(example_tr, 'A')
# xgdata.overall = len(xgdata.data)
# xgdata.split_to_kfold(xgdata.overall, 5)
# xgdata.Train('XGBOOST', xgdata.L[0][0], xgdata.L[0][1])
# xgdata.XG()
# print(xgdata.model.predict(xgb.DMatrix(np.array(example_te.drop(['A'],axis=1)))))

