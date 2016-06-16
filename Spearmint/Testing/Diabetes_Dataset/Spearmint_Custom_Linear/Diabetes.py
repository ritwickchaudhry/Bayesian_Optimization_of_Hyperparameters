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

Data =  Dataset.data
Target = Dataset.target


def svr(C_value, epsilon_value, linear_kernel_hyperparameters):

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

		Train_Features = np.divide(Train_Features,np.sqrt(np.power(linear_kernel_hyperparameters,10)))
		Test_Features = np.divide(Test_Features,np.sqrt(np.power(linear_kernel_hyperparameters,10)))

		############################################################################################################################
		################################            Linear with Different Length Scales      #######################################
		############################################################################################################################		
		gram_matrix_train = np.dot(Train_Features,Train_Features.T)
		gram_matrix_test = np.dot(Test_Features,Train_Features.T)
		model = SVR(kernel='precomputed', C=np.power(C_value,10)[0], epsilon=np.power(epsilon_value,10)[0])
		model.fit(gram_matrix_train, Train_Targets)
		predictions = model.predict(gram_matrix_test).tolist()
		sum += math.sqrt(mean_squared_error(predictions,Test_Targets))

	return sum/10
			


#For the Spearmint
def main(job_id,params):
	print "job_id : " + str(job_id)
	print params
	# result = svr(params['C_value'],params['gamma_value'],params['custom_rbf_hyperparameters'])
	result = svr(params['C'],params['epsilon'],params['custom_linear_hyperparameters'])	
	print result
	return result

# l = [1,1,1,1,1,1,1,1,1,1]
# print svr([1],[1],l)