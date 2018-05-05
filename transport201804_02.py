# **coding:UTF-8**
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_validation import KFold
from keras.layers.core import Dense
from keras.models import Sequential

example_trn = pd.read_csv('IO/test_after_n.csv')
example_ten = pd.read_csv('IO/test_after_n.csv')


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
        return L, kf

    def Train(self, model_name, tr_X, tr_Y):
        if model_name == 'SVR':
            models = SVR(kernel='poly', degree=4, gamma=1.8)
            models.fit(tr_X, tr_Y)
            return models
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
            models.fit(tr_X, tr_Y, batch_size=100, epochs=2)
            return models
        else:
            print("Error Parameter! It must be 'SVR' or 'NN'. " )

    def printResult(self, model_name, models,tr_X, tr_Y, te_X, te_Y):
        if model_name == 'NN':
            result = models.evaluate(te_X, te_Y)
            print('\nNeural Network Training Finish & result')
            print('Test result loss:', result)
        tr_pred = models.predict(tr_X)
        te_pred = models.predict(te_X)

        def evaluation(Y, pred, way):
            if way =='RMSE':
                print("Root Mean Squared Error: %.3f" % np.sqrt(mean_squared_error(Y, pred)))
            elif way == 'R2':
                print('R2 score: %.3f' % r2_score(Y,pred))
            elif way == 'MAD':
                print("Mean Absolute Deviation: %.3f" % MAD(Y, pred) + "\r" )
            else:
                print("Error Parameter! It must be 'RMSE','R2' or 'MAD'.")

        print('---' + model_name + ' Train---')
        evaluation(tr_Y, tr_pred, 'RMSE')
        evaluation(tr_Y, tr_pred, 'R2')
        evaluation(tr_Y, tr_pred, 'MAD')
        print('---' + model_name + ' Test---')
        evaluation(te_Y, te_pred, 'RMSE')
        evaluation(te_Y, te_pred, 'R2')
        evaluation(te_Y, te_pred, 'MAD')
        return te_Y, te_pred

    def train_test_print(self,  model_name, t): #, yt= [], yp=[]
        # self.y_true = yt
        # self.y_pred = yp
        self.model = self.Train(model_name, tr_X, tr_Y)
        print('\n----%d iteration of ' % (t + 1) + model_name + ' ----')
        te_y, te_pred = self.printResult(model_name, self.model, tr_X, tr_Y, te_X, te_Y)
        # self.y_true.append(te_y)
        # self.y_pred.append(te_pred)
        # return self.y_true, self.y_pred
        return te_y, te_pred

trdata = ReadData(example_trn, 'A')
trdata.overall = len(trdata.data)
trdata.L,  trdata.kf = trdata.split_to_kfold(trdata.overall, 5)
trdata.model = trdata.Train('NN', trdata.L[0][0],trdata.L[0][1])
trdata.D = {-1:([],[])}

for t, pred in enumerate(trdata.kf):
    tr_X, tr_Y, te_X, te_Y = trdata.L[t][0],trdata.L[t][1],trdata.L[t][2],trdata.L[t][3]
    trdata.D[t] = trdata.D[t-1].append(trdata.train_test_print('NN', t)) #, trdata.D[t-1][0], trdata.D[t-1][1]
    print(trdata.D[0][1])
print('1111111111111111')
print(trdata.D[4][1])

# print(pd.concat(pd.DataFrame([trdata.D[4][1]])))

