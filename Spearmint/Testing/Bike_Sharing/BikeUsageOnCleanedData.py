import pandas as pd
import numpy as np
import csv
import random
import math
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import statsmodels.api as stats


Data=pd.read_csv("Data.csv")

columns = Data.columns.tolist()

train = Data.loc[random.sample(list(Data.index),int(0.8 * len(Data.index)))]
Test_data = Data.loc[~Data.index.isin(train.index)]									

columns_1 = [columns_modified for columns_modified in columns if columns_modified not in ["casual", "registered", "cnt",  "April", "Wednesday", "Thursday", "October", "Season3"]]

columns_2 = [columns_modified for columns_modified in columns if columns_modified not in ["casual", "registered", "cnt", "February", "Saturday", "Thursday", "Wednesday" ,"August", "April", "May", "October", "July"]]

def svr(C_value,gamma_value):
	print "Yo"
	model = SVR(kernel='rbf', C=C_value[0], gamma=float(gamma_value[0]))
	model.fit(train[columns_1], train["casual"])
	predictions_casual = model.predict(Test_data[columns_1])
	# predictions_casual_train = model.predict(Data[columns_1])

	model = SVR(kernel='rbf', C=C_value[0], gamma=float(gamma_value[0]))
	model.fit(train[columns_2], train["registered"])
	predictions_registered = model.predict(Test_data[columns_2])
	# predictions_registered_train = model.predict(Data[columns_2])

	predictions = predictions_casual + predictions_registered
	predictions = predictions.tolist()
	# predictions_train = predictions_casual_train + predictions_registered_train

	for i in range(len(predictions)):
		if int(predictions[i]) < 0:
			predictions[i]=str(0)

	return math.sqrt(mean_squared_error(predictions,Test_data["cnt"]))



#For the Spearmint
def main(job_id,params):
	print "job_id : " + str(job_id)
	print params
	result = svr(params['C_value'],params['gamma_value'])
	print result
	return result