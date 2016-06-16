language: PYTHON
name:     "Diabetes"

variable {
	name: "custom_rbf_hyperparameters"
	type: FLOAT
	size: 10
	min: -2
	max: 3
}

variable {
	name: "C"
	type: FLOAT
	size: 1
	min: 0
	max: 10
}

variable {
	name: "gamma"
	type: FLOAT
	size: 1
	min: 0
	max: 10
}

variable {
	name: "degree"
	type: INT
	size: 1
	min: 1
	max: 5
}

variable {
	name: "kernel"
	type: ENUM
	size: 1
	options: "custom_rbf"
	options: "rbf"
	options: "poly"
	options: "linear"
	options: "sigmoidal"
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


