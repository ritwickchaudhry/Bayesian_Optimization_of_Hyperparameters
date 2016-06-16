import sys
import os
import csv
import math
import matplotlib.pyplot as plotter
import random
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as stats
from sklearn import svm
from sklearn.svm import SVR
# from sklearn.cross_validation import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
import pandas as pd

data = []
listx = []
cnt = []

####################################################################
#-------------------------READ TRAINING DATA-----------------------#
####################################################################

input=open("bikeDataTrainingUpload.csv","r")

lines = input.readlines()

out_lines = []
header = []
j=0

for line in lines:
	if j is not 0:
		line = line.split(',')
		line[-1] = line[-1][:-1]
		###########################################################
		#-------------------------------MONTH---------------------#
		###########################################################
		month = int(line[2])
		for i in range (1,12):
			if i is not month:
				line.append("0")
			else:
				line.append("1")
		###########################################################
			# if i is 12:
			# 	line[-1] = line[-1] + "\n"
		# print ("-------")
		# print line
		# print ("-------")
		###########################################################
		#-------------------------SEASON--------------------------#
		###########################################################
		season = int(line[0])
		for i in range (1,4):
			if i is not season:
				line.append("0")
			else:
				line.append("1")
		###########################################################

		###########################################################
		#-------------------------WEEKDAY-------------------------#
		###########################################################
		weekday = int(line[4])
		for i in range (0,6):
			if i is not weekday:
				line.append("0")
			else:
				line.append("1")
		###########################################################

		###########################################################
		#---------------------WEATHER_SITUATION-------------------#
		###########################################################
		weather_sit = int(line[6])
		for i in range (1,3):
			if i is not weather_sit:
				line.append("0")
			else:
				line.append("1")
		###########################################################

		line = line[:6] + line[7:]
		line = line[:4] + line[5:]
		line = line[:2] + line[3:]
		line = line[1:]
		# print line
		out_lines.append(line)
	else:
		line  = ["yr","holiday", "workingday","temp","atemp","hum","windspeed","casual","registered","cnt", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "Season1", "Season2", "Season3","Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Weather1", "Weather2"]
		# print len(line)
		out_lines.append(line)

	# print line
	# print line
	j = j+1
input.close()


output=open("Data.csv","wb")
csv_writer = csv.writer(output)
csv_writer.writerows(out_lines)
output.close()

##################################################################



####################################################################
#-------------------------READ TESTING DATA-----------------------#
####################################################################

input=open("TestX.csv","r")

lines = input.readlines()

out_lines = []
header = []
j=0

for line in lines:
	if j is not 0:
		line = line.split(',')
		line[-1] = line[-1][:-1]
		###########################################################
		#-------------------------------MONTH---------------------#
		###########################################################
		month = int(line[2])
		for i in range (1,12):
			if i is not month:
				line.append("0")
			else:
				line.append("1")
		###########################################################
			# if i is 12:
			# 	line[-1] = line[-1] + "\n"
		# print ("-------")
		# print line
		# print ("-------")
		###########################################################
		#-------------------------SEASON--------------------------#
		###########################################################
		season = int(line[0])
		for i in range (1,4):
			if i is not season:
				line.append("0")
			else:
				line.append("1")
		###########################################################

		###########################################################
		#-------------------------WEEKDAY-------------------------#
		###########################################################
		weekday = int(line[4])
		for i in range (0,6):
			if i is not weekday:
				line.append("0")
			else:
				line.append("1")
		###########################################################

		###########################################################
		#---------------------WEATHER_SITUATION-------------------#
		###########################################################
		weather_sit = int(line[6])
		for i in range (1,3):
			if i is not weather_sit:
				line.append("0")
			else:
				line.append("1")
		###########################################################

		line = line[:6] + line[7:]
		line = line[:4] + line[5:]
		line = line[:2] + line[3:]
		line = line[1:]
		# print line
		out_lines.append(line)
	else:
		line  = ["yr","holiday", "workingday","temp","atemp","hum","windspeed", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "Season1", "Season2", "Season3","Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Weather1", "Weather2"]
		# print len(line)
		out_lines.append(line)

	# print line
	# print line
	j = j+1
input.close()

output=open("Testing.csv","wb")
csv_writer = csv.writer(output)
csv_writer.writerows(out_lines)
output.close()

##################################################################


Data=pd.read_csv("Data.csv")
Test_Data=pd.read_csv("Testing.csv")
# Data.columns=header
# print Data

# target = Data["atemp"]
# target2 = target
# target=[x*x*x for x in target]
# target2=[x*x for x in target2]
# Data = Data.drop("atemp", axis=1)

# Data["atemp"] = np.array(target)
# Data["atemp2"] = np.array(target2)



#print Data.columns
#print Data.shape
columns = Data.columns.tolist()
print columns


# train = Data.sample(frac=0.8, random_state=1)

train = Data.loc[random.sample(list(Data.index),int(0.8 * len(Data.index)))]
test_data = Data.loc[~Data.index.isin(train.index)]									

# columns_1 = [columns_modified for columns_modified in columns if columns_modified not in ["casual", "registered", "cnt",  "January", "June", "July", "February", "Season1", "Season3"]]
columns_1 = [columns_modified for columns_modified in columns if columns_modified not in ["casual", "registered", "cnt",  "April", "Wednesday", "Thursday", "October", "Season3"]]

# columns_2 = [columns_modified for columns_modified in columns if columns_modified not in ["casual", "registered", "cnt", "February", "January", "Saturday", "Wednesday", "August", "April", "June", "October"]]
columns_2 = [columns_modified for columns_modified in columns if columns_modified not in ["casual", "registered", "cnt", "February", "Saturday", "Thursday", "Wednesday" ,"August", "April", "May", "October", "July"]]

list_it = np.arange(0, 2, 0.1)
# print len(list_it)
# print list_it
# for i in range()

# out = open("alpha.txt", "wb")
'''

min_i = 0
min_error=1000000000

for i in list_it:

	# model = RidgeCV(alphas=[i], cv=5)
	# # model = LinearRegression()
	# model.fit(train[columns], train["cnt"])

	# predictions_cnt = model.predict(test_data[columns])
	#print prediction

	# 	print i
	# model = RidgeCV(alphas=[i], cv=5)
	model = SVR(kernel='rbf', C=1e4*37/38, gamma=i)
	model.fit(Data[columns], Data["casual"])
	predictions_casual = model.predict(Data[columns])


	# model = RidgeCV(alphas=[i], cv=5)
	model = SVR(kernel='rbf', C=1e4*37/38, gamma=i)
	model.fit(Data[columns_2], Data["registered"])
	predictions_registered = model.predict(Data[columns_2])



	predictions = predictions_casual + predictions_registered

	# out.write(str(i)+" ")

	print( str(math.sqrt(mean_squared_error(predictions, Data["cnt"])) )  + "\n")
	err = math.sqrt(mean_squared_error(predictions, Data["cnt"]))
	if err < min_error:
		min_error = err
		min_i = i



print "------------ALPHA------------------"
print min_i
print "-----------------------------------"

print "------------ERROR------------------"
print min_error
print "-----------------------------------"


'''

# model = RidgeCV(alphas=[min_i], cv=5)
model = SVR(kernel='rbf', C=1e4, gamma=0.124)
model.fit(Data[columns_1], Data["casual"])
predictions_casual = model.predict(Test_Data[columns_1])
predictions_casual_train = model.predict(Data[columns_1])

# model = RidgeCV(alphas=[min_i], cv=5)
model = SVR(kernel='rbf', C=1e4, gamma=0.124)
model.fit(Data[columns_2], Data["registered"])
predictions_registered = model.predict(Test_Data[columns_2])
predictions_registered_train = model.predict(Data[columns_2])



predictions = predictions_casual + predictions_registered
predictions = predictions.tolist()
predictions_train = predictions_casual_train + predictions_registered_train
# print len(predictions)

# print predictions

print math.sqrt(mean_squared_error(predictions_train,Data["cnt"]))

for i in range(len(predictions)):
	if int(predictions[i]) < 0:
		predictions[i]=str(0)

count=0

final_output = open("output.csv","wb")
csv_writer2 = csv.writer(final_output)
# print predictions
csv_writer2.writerow(["id","cnt"])

for i in predictions:
	# print i
	csv_writer2.writerow([str(count),int(i)])
	count = count+1
final_output.close()

# print "####################"
# print columns_2
# print "####################"

fac = Data[columns_1]
tgt = Data["casual"]
fac = stats.add_constant(fac)
est = stats.OLS(tgt, fac).fit()
print est.summary()
print est.mse_resid


#plotter.savefig('Windspeed.png')
#print len(data)