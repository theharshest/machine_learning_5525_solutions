# Q.4

import sys
import numpy as np
import random
import scipy.sparse
import math

def load_dataset(data_filename, labels_filename):
	# Load dataset from file into numpy arrays
	data 	 = np.genfromtxt(data_filename, delimiter=",")
	targets  = np.genfromtxt(labels_filename, delimiter=",")

	# Subtracting 1 from the first column, so that document id can be used as index later
	data = np.hstack(((data[:, 0]-1).reshape(data.shape[0],1), data[:, 1:]))

	return data, targets

def words_cond_prob(n_words, data, targets):

	words_class_mat = np.zeros((n_words, 20))

	# Count of each word in each class, words X classes matrix
	for row in data:
		words_class_mat[row[1]-1, targets[row[0]]-1] += row[2]

	words_prob = np.zeros((n_words, 20))

	# Probability of each word in a given class
	for i in range(n_words):
		for j in range(20):
			words_prob[i, j] = float(words_class_mat[i, j])/(np.sum(words_class_mat[:, j]+1))

	# Adding a small amount to make all probabilities non-zero
	words_prob += 10**-10

	# Returning log of probabilities
	return np.log(words_prob)

def class_priors(targets, n_documents):
	priors = np.zeros(20)
	for i in range(20):
		priors[i] = targets[targets == i+1].shape[0]

	return priors/n_documents

def transform_data(n_documents, train_data, n_words):
	data = np.zeros((n_documents, n_words))

	for row in train_data:
		data[row[0], row[1]-1] += row[2]

	return data

def sigmoid(x):
	return 1/(1+math.exp(-x))

def zero_one_encoding(n_documents, targets):
	data = np.zeros((n_documents, 20))
	i = 0	
	for t in targets:
		data[i, t-1] = 1
		i += 1

	return data 

def logisticRegression(data_filename, labels_filename, num_splits, train_percent):
	# Loading dataset
	data, targets = load_dataset(data_filename, labels_filename)

	# Number of words and documents
	n_words 	  = np.unique(data[:, 1]).shape[0]
	n_documents   = np.unique(data[:, 0]).shape[0]

	logistic_regression_error_rates = []

	# Random num_splits 80-20 splits evaluation
	for ns in range(num_splits):

		train_set_percentage = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

		# Splitting dataset in 80-20%
		train_doc_num 		 = random.sample(range(n_documents), int(n_documents*train_percent))
		test_doc_num  		 = list( set(range(n_documents)) ^ set(train_doc_num))

		# Getting test data
		test_data 	  	 = np.vstack(([data[data[:,0] == i] for i in test_doc_num]))
		test_target   		 = targets[test_doc_num]

		# Error rates for each train_set_percentage
		error_rates 		 = []

		test_data_t = transform_data(n_documents, test_data, n_words)
		print "Iteration: " + str(ns+1)
		for percent in train_set_percentage:
			misclassification = 0

			# Getting percent documents from the train set
			n_train_documents = train_doc_num[:int(len(train_doc_num)*percent)]

			train_data 	  = np.vstack(([data[data[:,0] == i] for i in n_train_documents]))
			train_target 	  = targets[n_train_documents]

			# Initializing w with all ones
			w = np.ones((20, n_words))

			# Setting alpha to a small value
			alpha = 0.01

			train_data_t = transform_data(n_documents, train_data, n_words)

			it = 0
	
			targets_01 = zero_one_encoding(n_documents, targets)

			while it<100:
				# Calculating predicted classes from current w
				a = scipy.sparse.coo_matrix(w)
				b = scipy.sparse.coo_matrix(np.transpose(train_data_t))
				predicted_classes = (a*b).todense()

				# 0-1 encoding of classes
				for x in range(n_documents):
					ind = np.argmax(predicted_classes[:, x])
					predicted_classes[:, x] = 0
					predicted_classes[ind, x] = 1

				# Updating w
				a = scipy.sparse.coo_matrix(train_data_t)
				b = scipy.sparse.coo_matrix(predicted_classes-np.transpose(targets_01))
				grad = (b*a).todense()

				w = w - alpha*grad

				it+=1

			# Predicting on test data
			a = scipy.sparse.coo_matrix(w)
			b = scipy.sparse.coo_matrix(np.transpose(test_data_t))
			prediction = np.array((a*b).todense())

			for x in range(n_documents):
				ind = np.argmax(prediction[:, x])
				prediction[:, x] = 0
				prediction[ind, x] = 1

			for p in test_doc_num:
				if np.argmax(prediction[:, p]) != targets[p]-1:
					misclassification += 1

			error = float(misclassification)/len(test_doc_num)
			error_rates.append(error)

		print "Error Rates for current iteration:"
		print error_rates

		logistic_regression_error_rates.append(error_rates)

	print "Average Error Rates:"
	print np.mean(logistic_regression_error_rates, axis=0)
	print "Standard Deviation"
	print np.std(np.mean(logistic_regression_error_rates, axis=0))	

def naiveBayesDiscrete(data_filename, labels_filename, num_splits, train_percent):
	# Loading dataset
	data, targets = load_dataset(data_filename, labels_filename)

	# Number of words and documents
	n_words 	  = np.unique(data[:, 1]).shape[0]
	n_documents   = np.unique(data[:, 0]).shape[0]

	naive_bayes_error_rates = []

	# Random num_splits 80-20 splits evaluation
	for ns in range(num_splits):

		train_set_percentage = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

		# Splitting dataset in 80-20%
		train_doc_num 		 = random.sample(range(n_documents), int(n_documents*train_percent))
		test_doc_num  		 = list( set(range(n_documents)) ^ set(train_doc_num))

		# Getting test data
		test_data 	  		 = np.vstack(([data[data[:,0] == i] for i in test_doc_num]))
		test_target   		 = targets[test_doc_num]

		# Error rates for each train_set_percentage
		error_rates 		 = []

		print "Iteration: " + str(ns+1)
		for percent in train_set_percentage:
			misclassification = 0

			# Getting percent documents from the train set
			n_train_documents = train_doc_num[:int(len(train_doc_num)*percent)]

			train_data 		  = np.vstack(([data[data[:,0] == i] for i in n_train_documents]))
			train_target 	  = targets[n_train_documents]

			# Conditional probabilities
			words_prob 		  = words_cond_prob(n_words, train_data, targets)
			# Class prior probabilities
			priors 	   		  = class_priors(train_target, len(n_train_documents))

			for n in test_doc_num:
				# All rows for a document with index n
				doc 		= data[data[:, 0] == n]
				# Probabilities for all classes for document n
				final_probs = np.zeros(20)
				for row in doc:
					for i in range(20):
						# Log probablities add up - n11*log(P(w1|C1)) + n21*log(P(w2|C1)) + ...
						# where n11 is count of word 1 in document 1, n21 is count of word 2 in document 1
						final_probs[i] += row[2] * words_prob[row[1]-1, i]

				prediction = np.argmax(final_probs)
				true_class = targets[n]

				if prediction != true_class:
					misclassification += 1

			error = float(misclassification)/len(test_doc_num)
			error_rates.insert(0, error)

		print "Error Rates for current iteration:"
		print error_rates
		naive_bayes_error_rates.append(error_rates)

	print "Average Error Rates:"
	print np.mean(naive_bayes_error_rates, axis=0)

	print "Standard Deviation:"
	print np.std(np.mean(naive_bayes_error_rates, axis=0))

if __name__ == "__main__":
	data_filename 	= sys.argv[1]
	labels_filename = sys.argv[2]
	num_splits    	= int(sys.argv[3])
	train_percent 	= float(sys.argv[4])/100
	print "Running Logistic Regression:"
	logisticRegression(data_filename, labels_filename, num_splits, train_percent)
	print "Running Naive Bayes:"
	naiveBayesDiscrete(data_filename, labels_filename, num_splits, train_percent)
