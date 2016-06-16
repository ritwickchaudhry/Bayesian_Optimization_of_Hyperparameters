import math

def normal(data,mean,variance):
	# print data-mean
	return -1*(((data-mean)**2)/(2*variance)) - (0.5 * math.log(2*3.1415*variance))

a =  math.log(0.33333) + normal(67.7854,6.0998,13.5408)
b =  math.log(0.33333) + normal(67.7854,119.3287,9.4803)
c =  math.log(0.33333) + normal(67.7854,65.7801,12.6203)

d = math.exp(a) + math.exp(b) + math.exp(c)

print math.exp(a)
print math.exp(b)
print math.exp(c)

print math.exp(a)/d