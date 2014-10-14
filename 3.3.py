# Q.3 (iii)

import sys
import numpy as np

def load_dataset(filename):
	# Load dataset from file into numpy arrays
	data 	 = np.genfromtxt(filename, delimiter=",")
	targets  = data[:, 0]
	features = data[:, 1:]

	return targets, features

def convert_targets(filename, targets):
	# Convert target values from column vectors to bit-vector representation
	n_samples = targets.shape[0]
	if "spam" in filename:
		target = np.zeros((n_samples, 2))
		for i in range(n_samples):
			if targets[i] == 0:
				target[i, 0] = 1
			else:
				target[i, 1] = 1
	else:
		target = np.zeros((n_samples, 4))
		for i in range(n_samples):
			if targets[i] == 1:
				target[i, 0] = 1
			elif targets[i] == 3:
				target[i, 1] = 1
			elif targets[i] == 7:
				target[i, 2] = 1
			elif targets[i] == 8:
				target[i, 3] == 1

	return target

def linear_model(filename, targets, features):
	# Calculating three terms of the linear discriminant separately
	target_matrix 	 = np.transpose(convert_targets(filename, targets))
	X_pseudo_inverse = np.transpose(np.linalg.pinv(features))
	model 			 = np.dot(target_matrix, X_pseudo_inverse)

	return model

def SqClass(filename, num_crossval):
	targets, features = load_dataset(filename)
	n_samples 		  = targets.shape[0]

	features = np.hstack((np.ones((features.shape[0], 1)), features))

	i = 1
	j = n_samples/num_crossval

	test_error_rates  = []
	train_error_rates = []

	# 10-fold cross validation
	while i < n_samples:
		# Test Set
		test_set 	   = features[i:i+j, :]
		test_targets   = targets[i:i+j]
		# Train Set
		train_set 	   = np.vstack((features[:i, :], features[i+j+1:, :]))
		train_targets  = np.append(targets[:i], targets[i+j+1:])

		model 		   = linear_model(filename, train_targets, train_set)
		
		n_test_samples = test_targets.shape[0]
		n_train_samples = train_targets.shape[0]
		converted_test_targets = convert_targets(filename, test_targets)
		converted_train_targets = convert_targets(filename, train_targets)

		misclassifications = 0

		for k in range(n_test_samples):
			test_sample 	 = test_set[k, :]
			prediction  	 = np.dot(model, test_sample)

			prediction_class = np.argmax(prediction)

			true_class 		 = np.argmax(converted_test_targets[k, :])

			if prediction_class != true_class:
				misclassifications+=1

		error_rate = float(misclassifications)/n_test_samples

		test_error_rates.append(error_rate)

		misclassifications = 0

		for k in range(n_train_samples):
			train_sample 	 = train_set[k, :]
			prediction  	 = np.dot(model, train_sample)

			prediction_class = np.argmax(prediction)

			true_class 		 = np.argmax(converted_train_targets[k, :])

			if prediction_class != true_class:
				misclassifications+=1

		error_rate = float(misclassifications)/n_train_samples

		train_error_rates.append(error_rate)

		i += j

	print "Test set error rates:"
	print test_error_rates
	print "Standard Deviation:"
	print np.std(np.array(test_error_rates))

	print "Train set error rates:"
	print train_error_rates
	print "Standard Deviation:"
	print np.std(np.array(train_error_rates))

if __name__ == "__main__":
	filename 	 = sys.argv[1]
	num_crossval = int(sys.argv[2])
	SqClass(filename, num_crossval)