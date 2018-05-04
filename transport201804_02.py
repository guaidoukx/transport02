# **coding:UTF-8**
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_validation import KFold
from keras.layers.core import Dense
from keras.models import Sequential

example_tr = pd.read_csv('IO/train_after_n.csv')
example_te = pd.read_csv('IO/test_after_n.csv')

class ReadData(object):
    def __init__(self,data):
        self.data = data
