import time
import random
import sys
import math
import numpy as np

def load_dataset(filename):
	# Load dataset from file into numpy arrays
	data = np.genfromtxt(filename, delimiter=",")

	return data

def sumxy(samples):
	s = np.zeros(samples.shape[1] - 1)
	for samp in samples:
		s += samp[1:] * samp[0]

	return s

def svm(sample, w, raw=0):
	y = np.dot(w, sample[1:])

	# Return raw value of y if raw=1
	if raw:
		return y

	if y<0:
		return -1
	else:
		return 1

def objective_function(l, w, k, samples):
	total_loss = 0
	for sample in samples:
		total_loss += max(0, 1 - svm(sample, w, 1) * sample[0])
	return np.linalg.norm(w) * np.linalg.norm(w) / 2 + total_loss / k

def mysgdsvm(filename, k, numruns):
	# Loading dataset
	data = load_dataset(filename)

	# Initializing variables
	l = 0.5
	T = numruns
	n_features = data.shape[1] - 1
	n_samples = data.shape[0]

	# Making classes as 1 and -1
	for i in range(n_samples):
		if data[i][0] == 3:
			data[i][0] = -1

	w = np.zeros(n_features)
	total_time = 0
	obj_func_vals = []
	delta_time_vals = []

	for t in range(T):
		starttime = time.time()

		num_samples = range(n_samples)
		random.shuffle(num_samples, random.random)

		# Taking k random sample indices
		indices = []

		class_1 = 0
		class_2 = 0

		for ind in num_samples:
			if data[ind][0] == -1 and class_1 < k:
				indices.append(ind)
				class_1 += 1
				continue
			elif data[ind][0] == 1 and class_2 < k:
				indices.append(ind)
				class_2 += 1
				continue

		samples = data[indices, :]

		# Taking indices of misclassified samples
		num_samples = []

		for j in range(k):
			if svm(samples[j], w) != samples[j][0]:
				num_samples.append(j)

		samples = samples[num_samples, :]

		# Learning rate
		eta = 1/(l*(t+1))

		w_tmp = (1 - eta * l) * w + eta * sumxy(samples) / k + 1e-10

		w = min(1, 1/(math.sqrt(l) * np.linalg.norm(w_tmp))) * w_tmp

		endtime = time.time()

		obj_func = objective_function(l, w, k, data)
		delta_time = endtime - starttime

		total_time += delta_time
		obj_func_vals.append(obj_func)
		delta_time_vals.append(delta_time)

		print "Objective function value for the current iteration: " + str(obj_func)

	print "Average runtime: (in secs)" + str(total_time/numruns)
	print "Standard Deviation: " + str(np.std(delta_time_vals))

if __name__ == "__main__":
	filename = sys.argv[1]
	k = int(sys.argv[2])
	numruns = int(sys.argv[3])
	mysgdsvm(filename, k, numruns)