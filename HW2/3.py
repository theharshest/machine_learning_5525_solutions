import sys
import random
import time
import numpy as np

# To store errors
error_cache = 0

# Weight vector
w = np.zeros(100)

# Threshold
b = 0

eps = 0.00001

alphas = np.zeros(2000)

def load_dataset(filename):
	# Load dataset from file into numpy arrays
	data = np.genfromtxt(filename, delimiter=",")

	return data

def svm(i, data):
	# SVM prediction on this sample
	y = np.dot(w, data[i, 1:].T) - b

	if y<0:
		y = -1
	else:
		y = 1

	return y

def update_w(i, j, delta_alpha1, delta_alpha2, data):
	global w
	w = w + data[i][0] * delta_alpha1 * data[i, 1:] + data[j][0] * \
		delta_alpha2 * data[j, 1:]

def kernel(xi, xj):
	return np.dot(xi, xj.T)

def update_error_cache(i, j, delta_alpha1, delta_alpha2, delta_b, data, alphas, C, a1, a2):
	global error_cache

	# Updating alphas for non bound Lagrange multiplies
	if a1 != 0 and a1 != C:
		error_cache[i] = 0

	if a2 != 0 and a2 != C:
		error_cache[j] = 0

	for k in range(data.shape[0]):
		if k == i or k == j:
			continue
		if alphas[k] != 0 and alphas[k] != C:
			error_cache[k] = error_cache[k] + data[i][0] * delta_alpha1 * \
				kernel(data[i, 1:], data[k, 1:]) + data[j][0] * delta_alpha2 * \
					kernel(data[j, 1:], data[k, 1:]) + delta_b

def takestep(i, j, alphas, data, n_features, n_samples, C):
	global b

	if i == j:
		return 0

	alpha1 = alphas[i]
	alpha2 = alphas[j]

	y1 = data[i][0]
	y2 = data[j][0]

	# Error on sample 1
	e1 = svm(i, data) - y1
	e2 = svm(j, data) - y2
	error_cache[i] = e1

	s = y1 * y2

	# Calculating bounds of alpha1 and alpha2
	if y1 != y2:
		L = max(0, alpha2 - alpha1)
		H = min(C, C + alpha2 - alpha1)
	else:
		L = max(0, alpha2 + alpha1 - C)
		H = min(C, alpha2 + alpha1)

	if L == H:
		return 0

	# Kernels among two samples
	k11 = kernel(data[i, 1:], data[i, 1:])
	k12 = kernel(data[i, 1:], data[j, 1:])
	k22 = kernel(data[j, 1:], data[j, 1:])

	eta = 2*k12 - k11 - k22

	# Updating alpha2 as per value of eta, second derivative of objective function
	# As all input feature vectors are unique we don't need to consider the
	# case of eta = 0

	if eta < 0:
		a2 = alpha2 - y2 * (error_cache[i] - error_cache[j]) / eta
		if a2 < L:
			a2 = L
		elif a2 > H:
			a2 = H

	if a2 < 1e-8:
		a2 = 0
	elif a2 > C-1e-8:
		a2 = C

	if abs(a2 - alpha2) < eps * (a2 +  alpha2 + eps):
		return 0

	if a2 >= H:
		a2_clipped = H
	elif a2 > L and a2 < H:
		a2_clipped = a2
	else:
		a2_clipped = L

	# Calculating new alpha1 from clipped new alpha2
	a1 = alpha1 + s * (alpha2 - a2_clipped)

	b1 = 0
	b2 = 0

	# Flags to check if b1 or b1 is valid
	flag1 = 0
	flag2 = 0
	delta_b = b

	# Updating threshold if alpha1 not at bounds
	if a1 != 0 and a1 != C:
		b1 = e1 + y1 * (a1 - alpha1) * kernel(data[i, 1:], data[i, 1:]) + \
			y2 * (a2_clipped - alpha2) * kernel(data[i, 1:], data[j, 1:]) + b
		flag1 = 1

	# Updating threshold if alpha2 not at bounds
	if a2 != 0 and a2 != C:
		b2 = e2 + y1 * (a1 - alpha1) * kernel(data[i, 1:], data[i, 1:]) + \
			y2 * (a2_clipped - alpha2) * kernel(data[i, 1:], data[j, 1:]) + b
		flag2 = 1

	# Updating b
	if flag1 and flag2:
		b = b1
	elif (a1 == 0 or a1 == C) and (a2 == 0 or a2 == C) and L!=H:
		b = (b1 + b2) / 2
	elif flag1:
		b = b1
	else:
		b = b2

	delta_b = delta_b - b

	# Change in Lagrange parameters
	delta_alpha1 = a1 - alpha1
	delta_alpha2 = a2_clipped - alpha2

	# Updating w
	update_w(i, j, delta_alpha1, delta_alpha2, data)

	# Updating error cache
	update_error_cache(i, j, delta_alpha1, delta_alpha2, delta_b, data, alphas, C, a1, a2)

	# Updating alpha vector
	alphas[i] = a1
	alphas[j] = a2

	return 1

def check_kkt(j, data, alphas, tol, C, n_features, n_samples):
	global error_cache

	y2 = data[j][0]
	alpha2 = alphas[j]

	# Error on this sample
	e2 = svm(j, data) - y2
	error_cache[j] = e2

	r2 = e2 * y2	

	# Non-zero non C alphas
	nonalphas = [alpha for alpha in alphas if alpha !=0 and alpha != C]

	# Checking KKT conditions if this sample is incorrectly classified
	# i.e. stays out of tolerance
	if (r2 < -1*tol and alpha < C) or (r2 > tol and alpha > 0):
		if len(nonalphas) > 1:
			# Index of second lagrange multiplier
			i = 0
			# Searching for alpha which gives maximum step size approximated
			# by |e1 - e2|
			if e2>=0:
				i = np.argmin(error_cache)
			else:
				i = np.argmax(error_cache)
			if takestep(i, j, alphas, data, n_features, n_samples, C):
				return 1
		# For all not on bound alphas
		tmp1 = range(n_samples)
		random.shuffle(tmp1, random.random)
		for k in tmp1:
			if alphas[k] != 0 and alphas[k] != C:
				if takestep(k, j, alphas, data, n_features, n_samples, C):
					return 1
		# For all alphas possible
		tmp2 = range(n_samples)
		random.shuffle(tmp2, random.random)
		for k in tmp2:
			if takestep(k, j, alphas, data, n_features, n_samples, C):
					return 1

	return 0

def objective_function(alphas, kernel_mat, diagy):
	return np.sum(alphas) - 0.5 * (np.dot(alphas.T, np.dot(diagy, np.dot(kernel_mat, np.dot(diagy, alphas)))))

def mysmosvm(filename, numruns):
	global error_cache

	data = load_dataset(filename)

	# Number of samples and features
	n_samples = data.shape[0]
	n_features = data.shape[1] - 1

	# Making classes as 1 and -1
	for i in range(n_samples):
		if data[i][0] == 3:
			data[i][0] = -1

	# Initializing the lagrange parameters, threshold, C, error cache and
	# tolerance for Loose KKT conditions
	alphas = np.zeros(n_samples)
	C = 10
	error_cache = np.zeros(n_samples)
	tol = 0.00001

	for i in range(n_samples):
		error_cache[i] = svm(i, data) - data[i][0]

	# Kernel matrix, to be used for calculating the dual objective function
	kernel_mat = np.zeros((n_samples, n_samples))

	for i in range(n_samples):
		for j in range(n_samples):
			kernel_mat[i][j] = kernel(data[i, 1:], data[j, 1:])

	# Diagonal matrix y from actual column vetor
	diagy = np.zeros((n_samples, n_samples))
	for i in range(n_samples):
		diagy[i][i] = data[i][0]

	# Keeps a count of how many lagrange parameters are violating the KKT
	# condition
	num_changed = 0

	# Boolean to check if we need to iterate through all samples in outer loop
	# or non-bound samples to get the first lagrange paramater. We would alternate
	# between these two to make calculations faster
	examine_all = 1

	obj_func_vals = []
	delta_time_vals = []
	total_time = 0

	for i in range(numruns):
		starttime = time.time()
		num_changed = 0

		if examine_all:
			for j in range(n_samples):
				num_changed += check_kkt(j, data, alphas, tol, C, n_features, n_samples)
		else:
			for j in range(n_samples):
				alpha = alphas[j]
				if not alpha and alpha != C:
					num_changed += check_kkt(j, data, alphas, tol, C, n_features, n_samples)

		if examine_all == 1:
			examine_all = 0
		elif num_changed == 0:
			examine_all = 1

		endtime = time.time()

		obj_func = objective_function(alphas, kernel_mat, diagy)

		obj_func_vals.append(obj_func)
		delta_time = endtime - starttime

		total_time += delta_time
		delta_time_vals.append(delta_time)

		print "Objective function value for the current iteration: " + str(obj_func)

		i+=1

	print "Average runtime: " + str(total_time/numruns)
	print "Standard Deviation: " + str(np.std(delta_time_vals))

if __name__ == "__main__":
	filename = sys.argv[1]
	numruns = int(sys.argv[2])
	mysmosvm(filename, numruns)