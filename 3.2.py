# Q.3 (ii)

import sys
import numpy as np
from scipy import linalg

def load_dataset(filename):
	# Load dataset from file into numpy arrays
	data 	 = np.genfromtxt(filename, delimiter=",")
	targets  = data[:, 0]
	features = data[:, 1:]

	return targets, features

def calculate_prior_probs(filename, targets):
	n = targets.shape[0]
	if "spam" in filename:
		n1 = targets[targets == 0].shape[0]
		n2 = targets[targets == 1].shape[0]
		n1 = float(n1)/n
		n2 = float(n2)/n

		return [n1, n2]
	else:
		n1 = targets[targets == 1].shape[0]
		n2 = targets[targets == 3].shape[0]
		n3 = targets[targets == 7].shape[0]
		n4 = targets[targets == 8].shape[0]
		n1 = float(n1)/n
		n2 = float(n2)/n
		n3 = float(n3)/n
		n4 = float(n4)/n

		return [n1, n2, n3, n4]

def calculate_conditional_params(filename, projected_samples, train_targets):
	projected_samples = np.transpose(projected_samples)

	if "spam" in filename:
		# Samples in two classes
		s1 = projected_samples[train_targets == 0]
		s2 = projected_samples[train_targets == 1]

		# Mean of two classes
		m1 = np.mean(s1)
		m2 = np.mean(s2)

		# Variance of two classes
		v1 = np.var(s1)
		v2 = np.var(s2)

		return ([m1, v1], [m2, v2])
	else:
		# Samples in four classes
		s1 = projected_samples[train_targets == 1]
		s2 = projected_samples[train_targets == 3]
		s3 = projected_samples[train_targets == 7]
		s4 = projected_samples[train_targets == 8]

		# Mean of four classes
		m1 = np.mean(s1, axis=0)
		m2 = np.mean(s2, axis=0)
		m3 = np.mean(s3, axis=0)
		m4 = np.mean(s4, axis=0)

		# Covariance of four classes
		s1 = s1-m1
		s2 = s2-m2
		s3 = s3-m3
		s4 = s4-m4
		v1 = np.dot(np.transpose(s1), s1)
		v2 = np.dot(np.transpose(s2), s2)
		v3 = np.dot(np.transpose(s3), s3)
		v4 = np.dot(np.transpose(s4), s4)

		return ([m1, v1], [m2, v2], [m3, v3], [m4, v4])

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

def within_class_matrix(filename, train_targets, train_set):
	class_samples = []
	means 		  = []

	if "spam" in filename:
		n1 = train_set[train_targets == 0]
		n2 = train_set[train_targets == 1]

		m1 = np.mean(n1, axis=0)
		m2 = np.mean(n2, axis=0)

		class_samples.append(n1); class_samples.append(n2);
		means.append(m1); means.append(m2)
		n_features = train_set.shape[1]

		mat = np.zeros((n_features, n_features))

		n_classes = len(means)

		for i in range(n_classes):
			samples = class_samples[i] - means[i]
			for sample in samples:
				sample = sample.reshape(1, n_features)
				tmp = np.dot(np.transpose(sample), sample)
				mat = np.add(mat, tmp)

		return mat, np.transpose(means[0]), np.transpose(means[1])
	else:
		n1 = train_set[train_targets == 1]
		n2 = train_set[train_targets == 3]
		n3 = train_set[train_targets == 7]
		n4 = train_set[train_targets == 8]

		m1 = np.mean(n1, axis=0)
		m2 = np.mean(n2, axis=0)
		m3 = np.mean(n3, axis=0)
		m4 = np.mean(n4, axis=0)

		class_samples.append(n1); class_samples.append(n2); \
			class_samples.append(n3); class_samples.append(n4);
		means.append(m1); means.append(m2); means.append(m3); means.append(m4)
		n_features = train_set.shape[1]

		within = np.zeros((n_features, n_features))

		n_classes = len(means)

		for i in range(n_classes):
			samples = class_samples[i] - means[i]
			for sample in samples:
				sample = sample.reshape(1, n_features)
				tmp = np.dot(np.transpose(sample), sample)
				within = np.add(within, tmp)

		m = n1.shape[0]*m1 + n2.shape[0]*m2 + n3.shape[0]*m3 + n4.shape[0]*m4
		m = m/float(train_set.shape[0])

		between = np.zeros((n_features, n_features))

		for i in range(n_classes):
			mean = means[i] - m
			mean = mean.reshape(1, n_features)
			between = np.add(between, class_samples[i].shape[0]*np.dot(np.transpose(mean), mean))

		return within, between, np.transpose(means[0]), np.transpose(means[1]), \
			np.transpose(means[2]), np.transpose(means[3])

def fisher_discriminant(filename, train_targets, train_set):
	if "spam" in filename:
		within_class, m1, m2 = within_class_matrix(filename, \
								train_targets, train_set)

		inverse_within_class = linalg.inv(within_class)
		between_class		 = np.subtract(m1, m2)

		direction 			 = np.dot(inverse_within_class, between_class)
		magnitude 			 = np.linalg.norm(direction)
		direction 			 = direction/magnitude

		return direction
	else:
		within_class, between_class, m1, m2, m3, m4 = within_class_matrix(filename, \
								train_targets, train_set)

		inverse_within_class = np.linalg.pinv(within_class)

		result_mat 			 = np.dot(inverse_within_class, between_class)
		eigval, eigvec		 = np.linalg.eig(result_mat)

		idx = eigval.argsort()[-3:][::-1]
		eigvec = eigvec[:,idx]

		return eigvec

def diagFisher(filename, num_crossval):
	targets, features = load_dataset(filename)
	n_samples 		  = targets.shape[0]

	# Counter variables for cross-validation
	i 				  = 1
	j 				  = n_samples/num_crossval
	error_rate 		  = 0

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

		# LDA Direction
		direction	   = fisher_discriminant(filename, train_targets, train_set)

		if "spam" in filename:
			direction = direction.reshape(1, train_set.shape[1])
		else:
			direction = np.transpose(direction)

		# Projected samples from train set on the LDA direction
		projected_samples = np.dot(direction, np.transpose(train_set))
		projected_test_set = np.dot(direction, np.transpose(test_set))

		# Calculating prior and conditional probabilities for generative model \
		# (assuming Gaussian distribution of classes)
		prior_probs 	  = calculate_prior_probs(filename, train_targets)
		conditional_probs_params = calculate_conditional_params(filename, \
							projected_samples, train_targets)

		n_test_samples = test_targets.shape[0]
		n_train_samples = train_targets.shape[0]
		converted_test_targets = convert_targets(filename, test_targets)
		converted_train_targets = convert_targets(filename, train_targets)

		misclassifications = 0

		# Predicting on test set
		for k in range(n_test_samples):
			test_sample 	 = projected_test_set[:, k]

			if "spam" in filename:
				# Calculating posterior probabilities
				m1 = conditional_probs_params[0][0]
				m2 = conditional_probs_params[1][0]
				
				v1 = conditional_probs_params[0][1]
				v2 = conditional_probs_params[1][1]

				c1 = np.exp(-1 * (test_sample-m1)**2 / (2 * v1**2))/v1
				c2 = np.exp(-1 * (test_sample-m2)**2 / (2 * v2**2))/v2

				if c1>c2:
					prediction_class = 0
				else:
					prediction_class = 1
			else:
				# Calculating posterior probabilities
				m1 = conditional_probs_params[0][0]
				m2 = conditional_probs_params[1][0]
				m3 = conditional_probs_params[2][0]
				m4 = conditional_probs_params[3][0]

				v1 = conditional_probs_params[0][1]
				v2 = conditional_probs_params[1][1]
				v3 = conditional_probs_params[2][1]
				v4 = conditional_probs_params[3][1]

				test_sample = test_sample.reshape(1, 3)

				c1 = (-1 * np.dot(np.dot((test_sample-m1), np.linalg.inv(v1)), np.transpose(test_sample-m1)))/np.sqrt(np.linalg.det(v1))
				c2 = (-1 * np.dot(np.dot((test_sample-m2), np.linalg.inv(v2)), np.transpose(test_sample-m2)))/np.sqrt(np.linalg.det(v2))
				c3 = (-1 * np.dot(np.dot((test_sample-m3), np.linalg.inv(v3)), np.transpose(test_sample-m3)))/np.sqrt(np.linalg.det(v3))
				c4 = (-1 * np.dot(np.dot((test_sample-m4), np.linalg.inv(v4)), np.transpose(test_sample-m4)))/np.sqrt(np.linalg.det(v4))

				prediction_class = np.argmax(np.array([c1, c2, c3, c4]))

			true_class 		 = np.argmax(converted_test_targets[k, :])

			if prediction_class != true_class:
				misclassifications+=1

		error_rate = float(misclassifications)/n_test_samples

		test_error_rates.append(error_rate)

		misclassifications = 0

		# Predicting on train set
		for k in range(n_train_samples):
			train_sample 	 = projected_samples[:, k]

			if "spam" in filename:
				# Calculating posterior probabilities
				m1 = conditional_probs_params[0][0]
				m2 = conditional_probs_params[1][0]
				
				v1 = conditional_probs_params[0][1]
				v2 = conditional_probs_params[1][1]

				c1 = np.exp(-1 * (train_sample-m1)**2 / (2 * v1**2))/v1
				c2 = np.exp(-1 * (train_sample-m2)**2 / (2 * v2**2))/v2

				if c1>c2:
					prediction_class = 0
				else:
					prediction_class = 1
			else:
				# Calculating posterior probabilities
				m1 = conditional_probs_params[0][0]
				m2 = conditional_probs_params[1][0]
				m3 = conditional_probs_params[2][0]
				m4 = conditional_probs_params[3][0]

				v1 = conditional_probs_params[0][1]
				v2 = conditional_probs_params[1][1]
				v3 = conditional_probs_params[2][1]
				v4 = conditional_probs_params[3][1]

				train_sample = train_sample.reshape(1, 3)

				c1 = (-1 * np.dot(np.dot((train_sample-m1), np.linalg.inv(v1)), np.transpose(train_sample-m1)))/np.sqrt(np.linalg.det(v1))
				c2 = (-1 * np.dot(np.dot((train_sample-m2), np.linalg.inv(v2)), np.transpose(train_sample-m2)))/np.sqrt(np.linalg.det(v2))
				c3 = (-1 * np.dot(np.dot((train_sample-m3), np.linalg.inv(v3)), np.transpose(train_sample-m3)))/np.sqrt(np.linalg.det(v3))
				c4 = (-1 * np.dot(np.dot((train_sample-m4), np.linalg.inv(v4)), np.transpose(train_sample-m4)))/np.sqrt(np.linalg.det(v4))

				prediction_class = np.argmax(np.array([c1, c2, c3, c4]))

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
	diagFisher(filename, num_crossval)