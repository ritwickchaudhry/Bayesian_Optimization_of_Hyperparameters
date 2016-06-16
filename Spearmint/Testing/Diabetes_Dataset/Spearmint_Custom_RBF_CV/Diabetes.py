import pandas as pd
import numpy as np
import csv
import random
import math
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import statsmodels.api as stats
import sklearn.datasets as datasets
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler

Dataset = datasets.load_diabetes()
# columns = Data.columns.tolist()

Data =  Dataset.data
Target = Dataset.target

# print Data.shape
# print Target.shape


# print Test_Targets.shape

# def svr(C_value,gamma_value,kernel_hyperparameters):
def svr(kernel_hyperparameters):

	kf = KFold(Data.shape[0], n_folds=10, shuffle=True, random_state=0)
	sum = 0
	for Train_Index, Test_Index in kf:

		# SPLIT THE DATA ACCORDING TO THE INDEX SPLIT

		Scaler = StandardScaler()
		Train_Features = Scaler.fit_transform(Data[Train_Index])
		Test_Features = Scaler.transform(Data[Test_Index])
		
		Train_Targets = Target[Train_Index]
		Test_Targets = Target[Test_Index]

		gram_matrix_train = np.zeros((len(Train_Features),len(Train_Features)))
		for x in range(len(Train_Features)):
			for y in range(len(Train_Features)):
				sum_array = np.divide(np.square(Train_Features[x] - Train_Features[y]),np.square(np.power(kernel_hyperparameters,10))*2)
				gram_matrix_train[x][y] = np.exp(-1 * np.sum(sum_array))
		
		gram_matrix_test = np.zeros((len(Test_Features),len(Train_Features)))
		for x in range(len(Test_Features)):
			for y in range(len(Train_Features)):
				sum_array = np.divide(np.square(Test_Features[x] - Train_Features[y]),np.square(np.power(kernel_hyperparameters,10))*2)
				gram_matrix_test[x][y] = np.exp(-1 * np.sum(sum_array))
		

		model = SVR(kernel='precomputed')
		model.fit(gram_matrix_train, Train_Targets)
		predictions = model.predict(gram_matrix_test).tolist()
		sum += math.sqrt(mean_squared_error(predictions,Test_Targets))

	return sum/10


#For the Spearmint
def main(job_id,params):
	print "job_id : " + str(job_id)
	print params
	# result = svr(params['C_value'],params['gamma_value'],params['kernel_hyperparameters'])
	result = svr(params['kernel_hyperparameters'])	
	print result
	return result

# l = [1,1,1,1,1,1,1,1,1,1]
# svr(l)