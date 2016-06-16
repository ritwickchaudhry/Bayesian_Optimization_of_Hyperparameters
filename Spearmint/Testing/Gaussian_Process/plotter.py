import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pylab

number_of_points = 100

Domain_Array = np.linspace(0, 10, num=number_of_points)

Function_Values = []

file = open("Data.txt", 'r')
for line in file:
	Function_Values.append(float(line))
file.close()

Function_Values = np.array(Function_Values)
# figure
plt.plot(Domain_Array, Function_Values)
plt.plot((8.599854,8.599854),(-0.67125,0),'r-')
plt.show()


