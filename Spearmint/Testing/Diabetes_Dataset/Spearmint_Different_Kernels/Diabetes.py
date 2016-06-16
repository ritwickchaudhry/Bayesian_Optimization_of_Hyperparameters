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


Dataset = datasets.load_diabetes()

Data =  Dataset.data
Target = Dataset.target


def svr(C_value, gamma_value, degree_value, custom_rbf_hyperparameters, kernel):

	kf = KFold(Data.shape[0], n_folds=8, shuffle=False, random_state=None)
	# print kf.size
	sum = 0
	for Train_Index, Test_Index in kf:
		print "Enter"
		Train_Features = Data[Train_Index]
		Test_Features = Data[Test_Index]
		Train_Targets = Target[Train_Index]
		Test_Targets = Target[Test_Index]

		if kernel[0] == "custom_rbf":

			############################################################################################################################
			###########################            RBF with Different Length Scales      ###############################################
			############################################################################################################################		
			gram_matrix_train = np.zeros((len(Train_Features),len(Train_Features)))
			for x in range(len(Train_Features)):
				for y in range(len(Train_Features)):
					sum_array = np.divide(np.square(Train_Features[x] - Train_Features[y]),np.square(np.exp(custom_rbf_hyperparameters))*2)
					gram_matrix_train[x][y] = np.exp(-1 * np.sum(sum_array))

			
			gram_matrix_test = np.zeros((len(Test_Features),len(Train_Features)))
			for x in range(len(Test_Features)):
				for y in range(len(Train_Features)):
					sum_array = np.divide(np.square(Test_Features[x] - Train_Features[y]),np.square(np.exp(custom_rbf_hyperparameters))*2)
					gram_matrix_test[x][y] = np.exp(-1 * np.sum(sum_array))
			
			model = SVR(kernel='precomputed')
			model.fit(gram_matrix_train, Train_Targets)
			predictions = model.predict(gram_matrix_test).tolist()
			sum += math.sqrt(mean_squared_error(predictions,Test_Targets))

		if kernel[0] == 'rbf':

			############################################################################################################################
			#################################################        RBF      ##########################################################
			############################################################################################################################		
			
			model = SVR(kernel='rbf', C=C_value[0], gamma=gamma_value[0])
			model.fit(Train_Features, Train_Targets)
			predictions = model.predict(Test_Features).tolist()
			sum += math.sqrt(mean_squared_error(predictions,Test_Targets))
			metric = sum/8

		if kernel[0] == 'poly':

			############################################################################################################################
			############################################            Poly      ##########################################################
			############################################################################################################################		
			
			model = SVR(kernel='poly', degree=degree_value[0], gamma=gamma_value[0], C=C_value[0])
			model.fit(Train_Features, Train_Targets)
			predictions = model.predict(Test_Features).tolist()
			sum += math.sqrt(mean_squared_error(predictions,Test_Targets))

		if kernel[0] == 'linear':
			model = SVR(kernel='linear', gamma=gamma_value[0], C=C_value[0])
			model.fit(Train_Features, Train_Targets)
			predictions = model.predict(Test_Features).tolist()
			sum += math.sqrt(mean_squared_error(predictions,Test_Targets))

		if kernel[0] == 'sigmoidal':

			############################################################################################################################
			#######################################            sigmoidal      ##########################################################
			############################################################################################################################	
			
			model = SVR(kernel='sigmoidal', C=C_value[0], gamma=gamma_value[0])
			model.fit(Train_Features, Train_Targets)
			predictions = model.predict(Test_Feature).tolist()
			sum += math.sqrt(mean_squared_error(predictions,Test_Targets))

	metric = sum/8
	return metric


#For the Spearmint
def main(job_id,params):
	print "job_id : " + str(job_id)
	print params
	# result = svr(params['C_value'],params['gamma_value'],params['custom_rbf_hyperparameters'])
	result = svr(params['C'], params['gamma'], params['degree'], params['custom_rbf_hyperparameters'], params['kernel'])	
	print result
	return result

# l = [1,1,1,1,1,1,1,1,1,1]
# svr(l)