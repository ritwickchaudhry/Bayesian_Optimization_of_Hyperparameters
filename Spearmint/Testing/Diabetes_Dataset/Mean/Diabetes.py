# import pandas as pd
import numpy as np
# import csv
# import random
import math
import sklearn.linear_model as linear
from sklearn.metrics import mean_squared_error
import statsmodels.api as stats
import sklearn.datasets as datasets
from sklearn.cross_validation import KFold

Dataset = datasets.load_diabetes()

Target = Dataset.target

# print Data.shape
# print Target.shape

def Mean():
	kf = KFold(Target.shape[0], n_folds=8, shuffle=False, random_state=None)
	sum = 0
	for Train_Index, Test_Index in kf:
		Mean = np.sum(Target[Test_Index])/len(Test_Index)
		Mean_Array = np.ones(len(Test_Index)) * Mean
		Error = math.sqrt(mean_squared_error(Mean_Array, Target[Test_Index]))
		sum += Error
	print sum/8

Mean()
