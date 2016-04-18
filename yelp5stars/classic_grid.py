from classic.classifiers import TextClassifier, NaiveBayesClassifier, SGDTextClassifier, \
    LogisticClassifier, SVMClassifier, PerceptronClassifier, RandomForestTextClassifier

import csv
import numpy as np

TRAIN_FILE = "../../yelp_review_full_csv/train.csv"
TEST_FILE = "../../yelp_review_full_csv/test.csv"

def parse_file(filepath):
    with open(filepath, "r") as f:
        csv_reader = csv.reader(f)
        labels = []
        texts = []
        for row in csv_reader:
            labels.append(int(row[0]) - 1)
            texts.append(row[1])
        return (texts, labels)

def shuffle_data(train_values, labels):
        combined_lists = zip(train_values, labels)
        np.random.shuffle(combined_lists)
        return zip(*combined_lists)

if __name__ == '__main__':

    print "Getting data in format texts / labels"
    (train_texts, train_labels) = shuffle_data(*parse_file(TRAIN_FILE))
    (test_texts, test_labels) = shuffle_data(*parse_file(TEST_FILE))

    print "Number of train elements: " + str(len(train_texts))
    print "Number of test elements: " + str(len(test_texts))

    # Simple bag of words with SGD
    sgd = SGDTextClassifier(train_texts, train_labels,
                            test_texts=test_texts, test_labels=test_labels,
                            compute_features=True)
    sgd.grid_search_cv(verbose=5, n_jobs=4)
    test_error = sgd.get_test_error()
    print "Test error in held out set: " + str(test_error)
    print "=" * 20

    # Now with bigrams too
    sgd = SGDTextClassifier(train_texts, train_labels, ngram_range=(1,2),
                            test_texts=test_texts, test_labels=test_labels,
                            compute_features=True)
    sgd.grid_search_cv(verbose=5, n_jobs=4)
    test_error = sgd.get_test_error()
    print "Test error in held out set: " + str(test_error)
    print "=" * 20

    # Simple bag of words with NB
    nb = NaiveBayesClassifier(train_texts, train_labels,
                              test_texts=test_texts, test_labels=test_labels)
    nb.set_bag_of_ngrams() # Also can compute bag of words manually
    nb.grid_search_cv(verbose=5, n_jobs=4)
    test_error = nb.get_test_error()
    print "Test error in held out set: " + str(test_error)
    print "=" * 20

    # Now with bigrams too
    nb = NaiveBayesClassifier(train_texts, train_labels, ngram_range=(1,2),
                              test_texts=test_texts, test_labels=test_labels)
    nb.set_bag_of_ngrams() # Also can compute bag of words manually
    nb.grid_search_cv(verbose=5, n_jobs=4)
    test_error = nb.get_test_error()
    print "Test error in held out set: " + str(test_error)
    print "=" * 20

    # Simple bag of words with Random Forest
    rf = RandomForestTextClassifier(train_texts, train_labels,
                                    test_texts=test_texts, test_labels=test_labels)
    rf.set_bag_of_ngrams() # We can compute bag of words manually
    rf.grid_search_cv(n_jobs=4, verbose=5)
    test_error = rf.get_test_error()
    print "Test error in held out set: " + str(test_error)
    print "=" * 20

    # Now with bigrams too
    rf2 = RandomForestTextClassifier(train_texts, train_labels, ngram_range=(1,2),
                             test_texts=test_texts, test_labels=test_labels,
                             compute_features=True)
    rf2.grid_search_cv(n_jobs=4, verbose=5)
    test_error = rf2.get_test_error()
    print "Test error in held out set: " + str(test_error)
    print "=" * 20

    # Simple bag of words with Support Vector Machines
    svm = SVMClassifier(train_texts, train_labels,
                       test_texts=test_texts, test_labels=test_labels,
                       compute_features=True)
    svm.grid_search_cv(n_jobs=4, verbose=5)
    test_error = svm.get_test_error()
    print "Test error in held out set: " + str(test_error)
    print "=" * 20

    svm = SVMClassifier(train_texts, train_labels, ngram_range=(1,2),
                       test_texts=test_texts, test_labels=test_labels,
                       compute_features=True)
    svm.grid_search_cv(n_jobs=4, verbose=5)
    test_error = svm.get_test_error()
    print "Test error in held out set: " + str(test_error)
    print "=" * 20

    # Simple bag of words with a logistic classifier
    lr = LogisticClassifier(train_texts, train_labels,
                            test_texts=test_texts, test_labels=test_labels,
                            compute_features=True)
    lr.grid_search_cv(verbose=5, n_jobs=4)
    test_error = lr.get_test_error()
    print "Test error in held out set: " + str(test_error)
    print "=" * 20

    lr = LogisticClassifier(train_texts, train_labels, ngram_range=(1,2),
                            test_texts=test_texts, test_labels=test_labels,
                            compute_features=True)
    lr.grid_search_cv(verbose=5, n_jobs=4)
    test_error = lr.get_test_error()
    print "Test error in held out set: " + str(test_error)
    print "=" * 20

    # SGD up to 3-grams
    sgd = SGDTextClassifier(train_texts, train_labels, ngram_range=(1,3),
                            test_texts=test_texts, test_labels=test_labels,
                            compute_features=True)
    sgd.grid_search_cv(verbose=5, n_jobs=4)
    test_error = sgd.get_test_error()
    print "Test error in held out set: " + str(test_error)
    print "=" * 20