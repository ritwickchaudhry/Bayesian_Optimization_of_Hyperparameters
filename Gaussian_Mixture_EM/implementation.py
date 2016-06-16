import os
import sys
import math
import numpy as np
from scipy.misc import logsumexp
# import matplotlib.pyplot as plt

def normal(data,mean,variance):
	# print data-mean
	return -1*(((data-mean)**2)/(2*variance)) - (0.5 * math.log(2*3.1415*variance))

data_size = 10000

#----------------------------------------------------------------------
#--------------------Generate Data Randomly----------------------------
means = np.asarray([20,50,100]).reshape((-1,1))
variances = np.asarray([4,6,10]).reshape((-1,1))
coefficients = np.asarray([0.15,0.45,0.4]).reshape((-1,1)) 
# coefficients = np.asarray([1.0,0,0])

# print coefficients

data_1 = np.random.normal(means[0],math.sqrt(variances[0]),[data_size,1])
data_2 = np.random.normal(means[1],math.sqrt(variances[1]),[data_size,1])
data_3 = np.random.normal(means[2],math.sqrt(variances[2]),[data_size,1])

z = np.random.multinomial(1,coefficients.reshape(-1),data_size)
# print data_3.shape
# print "--------"
# print np.column_stack((data_1,data_2,data_3)).shape
# print "--------"
# print coefficients.shape
data = np.sum(np.column_stack((data_1,data_2,data_3)) * z,axis=1).reshape((-1,1))
# plt.hist(data)
# print data
# print data.shape

# sys.exit()

'''
print data_1
print data_2
print data_3
print np.column_stack((data_1,data_2,data_3))
print data
'''
#----------------------------------------------------------------------
#----------------------------------------------------------------------

estimated_means = np.random.uniform(low=-10.0,high=120.0,size=[3,1])
# print estimated_means.shape
print estimated_means

estimated_variances = np.random.uniform(low=0.0,high=20.0,size=[3,1])
# print estimated_variances.shape
print estimated_variances

mixing_coefficients = np.asarray([1.0/3,1.0/3,1.0/3]).reshape((-1,1))
# print mixing_coefficients

print data

def expectation(estimated_means,estimated_variances,mixing_coefficients):
	# print data.size()
	responsibilities = np.empty([data_size,3])
	temp_sum = np.empty([data_size,3])
	sum_column = np.zeros([data_size,1])
	# print "Yo1"
	# print sum_column.shape
	# print sum_column
	for i in range(3):
		# print normal(data,estimated_means[i],estimated_variances[i]).shape

		##### WORK HERE
		temp = math.log(mixing_coefficients[i]) + normal(data,estimated_means[i],estimated_variances[i])
		temp_sum[:,i] = temp.reshape(-1)
		#DONE
	# print mixing_coefficients
	# print normal(data,estimated_means[i],estimated_variances[i])
	# print mixing_coefficients[i]*normal(data,estimated_means[i],estimated_variances[i]) 
	# sum_column = sum_column + mixing_coefficients[i]*normal(data,estimated_means[i],estimated_variances[i]) 
	# sum_column = sum_column + math.log(mixing_coefficients) + normal(data,estimated_means[i],estimated_variance[i])
	# print "yooo"
	# print temp_sum.shape
	sum_column = logsumexp(temp_sum, axis=1)

	sum_column = sum_column.reshape((-1,1))
	# print sum_column
	# print sum_column.shape


	for i in range(3):
		# print np.divide(mixing_coefficients[i]*normal(data,estimated_means[i],estimated_variances[i]),sum_column)
		#WORK HERE
		# print "YOyo"
		# print normal(data,estimated_means[i],estimated_variances[i]).shape
		# print sum_column.shape
		responsibilities[:,i] = np.exp(math.log(mixing_coefficients[i]) + normal(data,estimated_means[i],estimated_variances[i]).reshape((-1,1)) - sum_column).reshape(-1)

		# responsibilities[:,i] = np.divide(mixing_coefficients[i]*normal(data,estimated_means[i],estimated_variances[i]),sum_column).reshape(-1)
		# print responsibilities

	return responsibilities
	# print "Shape :"
	# print responsibilities.shape()
	# print "---------------------"
	# return responsibilities


def maximization(responsibilities):
	print responsibilities
	sum_of_responsibilities = np.sum(responsibilities,axis=0).reshape((-1,1))
	print sum_of_responsibilities
	# print sum_of_responsibilities.shape
	# print sum_of_responsibilities
	# print np.dot(np.transpose(responsibilities),data).shape
	# print "yo"
	# print np.dot(np.transpose(responsibilities),data)
	# print sum_of_responsibilities
	# print data.shape
	next_estimated_means = np.divide(np.dot(np.transpose(responsibilities),data),sum_of_responsibilities)
	# print next_estimated_means
	# print next_estimated_means.shape
	next_estimated_variances = np.empty([3,1])
	for i in range(3):
		# print next_estimated_means[i].shape
		temp = (data - next_estimated_means[i])**2;
		next_estimated_variances[i,:] = np.dot(responsibilities[:,i],temp)/sum_of_responsibilities[i]

	next_mixing_coefficients = sum_of_responsibilities/data_size;
	return (next_estimated_means,next_estimated_variances,next_mixing_coefficients)
# def maximization(responsibilities):

# print expectation(estimated_means,estimated_variances,mixing_coefficients)
# print a.shape
# maximization(expectation(estimated_means,estimated_variances,mixing_coefficients))

def iterate():
	responsibilities = expectation(estimated_means,estimated_variances,mixing_coefficients)
	for i in range(100):
		(next_estimated_means,next_estimated_variances,next_mixing_coefficients) = maximization(responsibilities)
		print "Epochs : " + str(i)
		print "Estimated Mean : "
		# print data
		print next_estimated_means
		print "Estimated Variances : "
		print next_estimated_variances
		print "Estimated Mixing Coefficients : "
		print next_mixing_coefficients

		print "-------------------------------------------------------------"
		responsibilities = expectation(next_estimated_means,next_estimated_variances,next_mixing_coefficients)

iterate()
# a = np.asarray([1])
# print normal(a,0,3.1415)