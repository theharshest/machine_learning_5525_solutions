import numpy as np
import sys
import random
from math import log

def get_data(filename):
    data = np.genfromtxt(filename, delimiter=",", dtype=np.int64)
    return data

def attr_vals(filename, i):
    data = get_data(filename)
    return np.unique(data[:, i])

def get_bagging_sample(i, j, data):
    inds = [random.randint(i, j) for k in range(j-i)]
    return data[inds]

def get_entropy(data, attr):
    '''
    Calculates entropy for split at attr attribute
    '''
    entropy = 0
    values = np.unique(data[:, attr])
    for value in values:
        data_tmp = data[data[:, attr]==value]
        n1 = data_tmp[data_tmp[:, 0]==1].shape[0]
        n2 = data_tmp.shape[0] - n1
        p1 = n1/float(n1 + n2)
        p2 = 1 - p1

        if p1==0 or p2==0:
            continue

        entropy += -1 * (p1*log(p1, 2) + p2*log(p2, 2))

    return entropy

def decision_tree(filename, train_data, m):
    '''
    Build decision tree using IG as split criteria
    '''

    # Get M random features to split on
    n_features = random.sample(range(0, train_data.shape[1]-1), m)
    split2 = []
    attrs = []

    # Get entropies for all attributes and store the attribute for the giving
    # the lowest entropy
    entropies = [get_entropy(train_data, i+1) for i in n_features]
    split1 = np.argmin(entropies) + 1
    attrs.append(split1)

    # New datasets after the first split
    #ndatas = np.unique(train_data[:, split1])
    ndatas = attr_vals(filename, split1)
    datas = [train_data[train_data[:, split1]==i] for i in ndatas]

    split2 = []
    final_labels = []

    for data in datas:
        entropies = []
        for i in n_features:
            entropies.append(get_entropy(data, i+1))

        split22 = np.argmin(entropies) + 1
        split2.append(split22)
        attrs.append(split22)

        # New datasets after the second split
        #ndatas = np.unique(data[:, split22])
        ndatas = attr_vals(filename, split22)
        datas2 = [data[data[:, split22]==i] for i in ndatas]

        labels = []

        for dat in datas2:
            ones = dat[dat[:, 0]==1].shape[0]
            nones = dat[dat[:, 0]==-1].shape[0]

            # Assigning label based on majority of labels in leaf nodes
            if ones >= nones:
                labels.append(1)
            else:
                labels.append(-1)

        final_labels.append(labels)

    return split1, split2, final_labels

def get_error(classifiers, data):
    '''
    Calculates train/test error
    '''
    misclassification = 0

    for sample in data:
        pred = 0
        for classifier in classifiers:
            split1 = classifier[0]
            split2 = classifier[1]
            labels = classifier[2]

            ind1 = sample[split1]
            attr2 = split2[ind1 - 1]
            ind2 = sample[attr2]

            pred += labels[ind1-1][ind2-1]
        if (pred >=0 and sample[0] == -1) or (pred <0 and sample[0] == 1):
            misclassification += 1

    error = misclassification/float(data.shape[0])

    return error

def myRForest2(filename, M):
    '''
    Main random forest classifier function
    '''
    data = get_data(filename)

    n_samples = data.shape[0]

    all_errors = []

    for m in M:
        train_test_errors = []
        for i in range(10):
            # Train set lies between indices train_test_split1 and train_test_split2
            # Rest is Test set
            train_test_split1 = int(n_samples * 0.01 * i)
            train_test_split2 = int(n_samples * (0.9 + 0.01 * i))

            test_data1 = data[:train_test_split1, :]
            test_data2 = data[train_test_split2:, :]
            test_data = np.vstack((test_data1, test_data2))

            classifiers= []
            for j in range(100):
                # Getting final train set by random sampling from the original set
                train_data = get_bagging_sample(train_test_split1, train_test_split2, data)

                # Building and storing decision tree for the current sample set
                split1, split2, labels = decision_tree(filename, train_data, m)
                classifiers.append((split1, split2, labels))

            train_error = get_error(classifiers, train_data)
            test_error = get_error(classifiers, test_data)

            train_test_errors.append([train_error, test_error])
            all_errors.append([train_error, test_error])

        train_test_errors = np.array(train_test_errors)
        print "Number of features: " + str(m)
        print "Average Train Error: " + str(np.average(train_test_errors[:, 0]))
        print "Standard Deviation Train Error: " + str(np.std(train_test_errors[:, 0]))
        print "Average Test Error: " + str(np.average(train_test_errors[:, 1]))
        print "Standard Deviation Test Error: " + str(np.std(train_test_errors[:, 1]))

    all_errors = np.array(all_errors)

    print "Overall Average Train Error: " + str(np.average(all_errors[:, 0]))
    print "Overall Standard Deviation Train Error: " + str(np.std(all_errors[:, 0]))
    print "Overall Average Test Error: " + str(np.average(all_errors[:, 1]))
    print "Overall Standard Deviation Test Error: " + str(np.std(all_errors[:, 1]))

if __name__ == "__main__":
    filename = sys.argv[1]
    M = eval(sys.argv[2])
    myRForest2(filename, M)
