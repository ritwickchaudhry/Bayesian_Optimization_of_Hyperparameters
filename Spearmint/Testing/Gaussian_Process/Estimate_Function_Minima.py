import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import pylab

print "--------------"

number_of_points=100

Domain_Array = np.linspace(0,10,num=number_of_points,endpoint=True)

def create_data_to_file():
	Length_Scale_1 = 1.2
	Length_Scale_2 = 3
	Coefficient_1 = 0.3333
	Coefficient_2 = 0.6667
	Variance_Array = np.random.uniform(low=0.0001, high=0.001, size=number_of_points)
	# print Variance_Array
	# print np.diag(Variance_Array)
	# print len(Domain_Array)

	Covariance_Matrix = np.ones((number_of_points,number_of_points),dtype="float")

	Covariance_Matrix = Covariance_Matrix + np.diag(Variance_Array)

	for x in range(len(Domain_Array)):
		for y in range(len(Domain_Array)):
			Covariance_Matrix[x][y] = Coefficient_1 * np.exp(-1 * ((Domain_Array[x] - Domain_Array[y])**2) / (2 * (Length_Scale_1**2)) ) + Coefficient_2 * np.exp(-1 * ((Domain_Array[x] - Domain_Array[y])**2) / (2 * (Length_Scale_2**2)) )

	Function_Values = np.random.multivariate_normal(np.zeros((number_of_points),dtype="float"),Covariance_Matrix)

	file = open("Data.txt","wb")

	file.write("Domain_Array\n")

	file.writelines(["%s\n" % item  for item in Domain_Array])

	file.write("Function_Values\n")

	file.writelines(["%s\n" % item  for item in Function_Values])


	# print Domain

	plt.plot(Domain_Array, Function_Values)
	pylab.show()

# print Covariance_Matrix

#READ DATA FROM FILE
##############################
file = open("Data.txt",'r')
Function_Values = []
for line in file:
	Function_Values.append(float(line))
Function_Values = np.asarray(Function_Values)
file.close()
##############################


def find_nearest_below(x):
    index = np.searchsorted(Domain_Array, x, side="left")
    return index

def evaluate(x):
	print x
	lower_index = find_nearest_below(x);
	if lower_index == 99:
		return np.asarray(Function_Values[99]).reshape((-1,1))
	upper_index = lower_index + 1
	estimated_value = Function_Values[lower_index] + (x - Domain_Array[lower_index])*((Function_Values[upper_index] - Function_Values[lower_index])/(Domain_Array[upper_index] - Domain_Array[lower_index]))
	return estimated_value

#For the Spearmint
def main(job_id,params):
	print "job_id : " + str(job_id)
	print params
	# result = svr(params['C_value'],params['gamma_value'],params['custom_rbf_hyperparameters'])
	result = evaluate(params['x'])
	print result
	return result[0]


# print evaluate(3.84018980131)
# l = [1,1,1,1,1,1,1,1,1,1]
# print svr([1],[1],l)