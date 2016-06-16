language: PYTHON
name:     "Diabetes"

variable {
	name: "custom_linear_hyperparameters"
	type: FLOAT
	size: 10
	min: -4
	max: 4
}

variable {
	name: "C"
	type: FLOAT
	size: 1
	min: -4
	max: 4
}

variable {
	name: "epsilon"
	type: FLOAT
	size: 1
	min: -4
	max: 4
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


