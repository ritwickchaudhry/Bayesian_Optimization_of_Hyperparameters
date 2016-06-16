# import pandas as pd
import numpy as np
# import csv
# import random
import math
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import statsmodels.api as stats
import sklearn.datasets as datasets
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler

Dataset = datasets.load_diabetes()
# columns = Data.columns.tolist()

Data =  Dataset.data
Target = Dataset.target

# print Data.shape
# print Target.shape


def Linear_Regression():
	kf = KFold(Data.shape[0], n_folds=10, shuffle=True, random_state=0)
	# print kf.size
	sum = 0
	for Train_Index, Test_Index in kf:
		# print "Enter"
		Scaler = StandardScaler()
		Train_Features = Scaler.fit_transform(Data[Train_Index])
		# Train_Features = Data[Train_Index]
		# Test_Features = Data[Test_Index]
		Test_Features = Scaler.transform(Data[Test_Index])
		Train_Targets = Target[Train_Index]
		Test_Targets = Target[Test_Index]

		############################################################################################################################		      #######################################           Linear Regression          ###############################################
		############################################################################################################################		
		model = linear_model.LinearRegression()
		model.fit(Train_Features,Train_Targets)
		predictions = model.predict(Test_Features).tolist()
		sum += math.sqrt(mean_squared_error(predictions,Test_Targets))

	print sum/10

Linear_Regression()
