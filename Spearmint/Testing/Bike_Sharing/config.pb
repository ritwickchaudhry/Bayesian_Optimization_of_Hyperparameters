language: PYTHON
name:     "BikeUsageOnCleanedData"

variable {
 name: "C_value"
 type: FLOAT
 size: 1
 min:  0
 max:  2
}

variable {
	name: "gamma_value"
	type: ENUM
	size: 1
	options: "1e-1"
	options: "1e-2"
	options: "1e-3"
	options: "1e-4"
	options: "1e-5"
}

# Integer example
#
# variable {
#  name: "Y"
#  type: INT
#  size: 5
#  min:  -5
#  max:  5
# }

# Enumeration example
# 
# variable {
#  name: "Z"
#  type: ENUM
#  size: 3
#  options: "foo"
#  options: "bar"
#  options: "baz"
# }


