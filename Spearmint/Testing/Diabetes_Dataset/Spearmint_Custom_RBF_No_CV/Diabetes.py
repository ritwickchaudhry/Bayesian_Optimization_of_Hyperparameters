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
# columns = Data.columns.tolist()

Data =  Dataset.data
Target = Dataset.target

# print Data.shape
# print Target.shape

Train_Features = Data[:-50]
Test_Features = Data[-50:]

print Train_Features.shape
# print Test_Features.shape

Train_Targets = Target[:-50]
Test_Targets = Target[-50:]

print Train_Targets.shape
# print Test_Targets.shape44
'''
def custom_rbf(parameters):
	def my_kernel(x,y):
		gram_matrix = np.zeros((392,392))
		print "Started"
		for i in range (0,392):
			for j in range (0,392):
				sum_array = np.dot(np.square(x[i] - y[j]),np.reciprocal(np.square(parameters)/2))
				gram_matrix[i][j] = np.exp(-1 * np.sum(sum_array))
		# a =  np.exp(sum)
		# print a
		print "Done"
		return gram_matrix
	return my_kernel
'''

# def svr(C_value,gamma_value,kernel_hyperparameters):
def svr(kernel_hyperparameters):
	# model = SVR(kernel='rbf', C=C_value[0], gamma=float(gamma_value[0]))
	# func = custom_rbf(kernel_hyperparameters)
	gram_matrix_train = np.zeros((len(Train_Features),len(Train_Features)))
	print "Started"
	print len(Train_Features)
	for x in range(len(Train_Features)):
		# print x
		for y in range(len(Train_Features)):
			sum_array = np.divide(np.square(Train_Features[x] - Train_Features[y]),np.square(kernel_hyperparameters)*2)
			# print Train_Features[x]
			# print Train_Features[y]
			# print sum_array
			gram_matrix_train[x][y] = np.exp(-1 * np.sum(sum_array))
	# a =  np.exp(sum)
	# print a
	print "Done"
	
	gram_matrix_test = np.zeros((len(Test_Features),len(Train_Features)))
	print "Started"
	print len(Test_Features)
	for x in range(len(Test_Features)):
		# print x
		for y in range(len(Train_Features)):
			sum_array = np.divide(np.square(Test_Features[x] - Train_Features[y]),np.square(kernel_hyperparameters)*2)
			# print Train_Features[x]
			# print Train_Features[y]
			# print sum_array
			gram_matrix_test[x][y] = np.exp(-1 * np.sum(sum_array))
	# a =  np.exp(sum)
	# print a
	print "Done"
	

	model = SVR(kernel='precomputed')
	model.fit(gram_matrix_train, Train_Targets)
	predictions = model.predict(gram_matrix_test).tolist()

	return math.sqrt(mean_squared_error(predictions,Test_Targets))



#For the Spearmint
def main(job_id,params):
	print "job_id : " + str(job_id)
	print params
	# result = svr(params['C_value'],params['gamma_value'],params['kernel_hyperparameters'])
	result = svr(params['kernel_hyperparameters'])	
	print result
	return result

# l = [1,1,1,1,1,1,1,1,1,1]
# print svr([1e-1],[0.1],l)