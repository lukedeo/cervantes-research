import csv
import numpy as np

def shuffle_data(train_values, labels):
        combined_lists = zip(train_values, labels)
        np.random.shuffle(combined_lists)
        zipped = zip(*combined_lists)
        return list(zipped[0]), list(zipped[1])

##### AG News

def parse_ag_news(filepath):
    with open(filepath, "r") as f:
        csv_reader = csv.reader(f)
        labels = []
        texts = []
        for row in csv_reader:
            labels.append(int(row[0]) - 1)
            texts.append(row[1] + ".  " + row[2])
        return texts, labels


def get_ag_news_data():
    TRAIN_FILE = "../ag_news_csv/train.csv"
    TEST_FILE = "../ag_news_csv/test.csv"

    (train_texts, train_labels) = shuffle_data(*parse_ag_news(TRAIN_FILE))
    (test_texts, test_labels) = shuffle_data(*parse_ag_news(TEST_FILE))

    return train_texts, train_labels, test_texts, test_labels

##### Sogou

def parse_sogou(filepath):
    with open(filepath, "r") as f:
        import sys
        csv_reader = csv.reader(f)
        csv.field_size_limit(sys.maxsize)
        labels = []
        texts = []
        for row in csv_reader:
            labels.append(int(row[0]) - 1)
            texts.append(row[1] + ".  " + row[2])
        return texts, labels

def get_sogou_data():
    TRAIN_FILE = "../sogou_news_csv/train.csv"
    TEST_FILE = "../sogou_news_csv/test.csv"

    (train_texts, train_labels) = shuffle_data(*parse_sogou(TRAIN_FILE))
    (test_texts, test_labels) = shuffle_data(*parse_sogou(TEST_FILE))

    return train_texts, train_labels, test_texts, test_labels

##### DBpedia

def parse_dbpedia(filepath):
    with open(filepath, "r") as f:
        csv_reader = csv.reader(f)
        labels = []
        texts = []
        for row in csv_reader:
            labels.append(int(row[0]) - 1)
            texts.append(row[1] + ".  " + row[2])
        return texts, labels

def get_dbpedia_data():
    TRAIN_FILE = "../dbpedia_csv/train.csv"
    TEST_FILE = "../dbpedia_csv/test.csv"

    (train_texts, train_labels) = shuffle_data(*parse_dbpedia(TRAIN_FILE))
    (test_texts, test_labels) = shuffle_data(*parse_dbpedia(TEST_FILE))

    return train_texts, train_labels, test_texts, test_labels

##### Yelp 5 stars

def parse_yelp_full(filepath):
    with open(filepath, "r") as f:
        csv_reader = csv.reader(f)
        labels = []
        texts = []
        for row in csv_reader:
            labels.append(int(row[0]) - 1)
            texts.append(row[1])
        return texts, labels

def get_yelp_full_data():
    TRAIN_FILE = "../yelp_review_full_csv/train.csv"
    TEST_FILE = "../yelp_review_full_csv/test.csv"

    (train_texts, train_labels) = shuffle_data(*parse_yelp_full(TRAIN_FILE))
    (test_texts, test_labels) = shuffle_data(*parse_yelp_full(TEST_FILE))

    return train_texts, train_labels, test_texts, test_labels

##### Yelp polarity

def parse_yelp_polarity(filepath):
    with open(filepath, "r") as f:
        csv_reader = csv.reader(f)
        labels = []
        texts = []
        for row in csv_reader:
            labels.append(int(row[0]) - 1)
            texts.append(row[1])
        return texts, labels

def get_yelp_polarity_data():
    TRAIN_FILE = "../yelp_review_polarity_csv/train.csv"
    TEST_FILE = "../yelp_review_polarity_csv/test.csv"

    (train_texts, train_labels) = shuffle_data(*parse_yelp_polarity(TRAIN_FILE))
    (test_texts, test_labels) = shuffle_data(*parse_yelp_polarity(TEST_FILE))

    return train_texts, train_labels, test_texts, test_labels

##### Yahoo

def parse_yahoo(filepath):
    with open(filepath, "r") as f:
        csv_reader = csv.reader(f)
        labels = []
        texts = []
        for row in csv_reader:
            labels.append(int(row[0]) - 1)
            texts.append(row[1] + ".  " + row[2] + ". " + row[3])
        return texts, labels

def get_yahoo_data():
    TRAIN_FILE = "../yahoo_answers_csv/train.csv"
    TEST_FILE = "../yahoo_answers_csv/test.csv"

    (train_texts, train_labels) = shuffle_data(*parse_yahoo(TRAIN_FILE))
    (test_texts, test_labels) = shuffle_data(*parse_yahoo(TEST_FILE))

    return train_texts, train_labels, test_texts, test_labels

##### Amazon polarity

def parse_amazon_polarity(filepath):
    with open(filepath, "r") as f:
        csv_reader = csv.reader(f)
        labels = []
        texts = []
        for row in csv_reader:
            labels.append(int(row[0]) - 1)
            texts.append(row[1] + ". " + row[2])
        return texts, labels

def get_amazon_polarity_data():
    TRAIN_FILE = "../amazon_review_polarity_csv/train.csv"
    TEST_FILE = "../amazon_review_polarity_csv/test.csv"

    (train_texts, train_labels) = shuffle_data(*parse_amazon_polarity(TRAIN_FILE))
    (test_texts, test_labels) = shuffle_data(*parse_amazon_polarity(TEST_FILE))

    return train_texts, train_labels, test_texts, test_labels

##### Amazon full

def parse_amazon_full(filepath):
    with open(filepath, "r") as f:
        csv_reader = csv.reader(f)
        labels = []
        texts = []
        for row in csv_reader:
            labels.append(int(row[0]) - 1)
            texts.append(row[1] + ". " + row[2])
        return texts, labels

def get_amazon_full_data():
    TRAIN_FILE = "../amazon_review_full_csv/train.csv"
    TEST_FILE = "../amazon_review_full_csv/test.csv"

    (train_texts, train_labels) = shuffle_data(*parse_amazon_full(TRAIN_FILE))
    (test_texts, test_labels) = shuffle_data(*parse_amazon_full(TEST_FILE))

    return train_texts, train_labels, test_texts, test_labels

def test():

    a,b,c,d = get_ag_news_data()
    a,b,c,d = get_sogou_data()
    a,b,c,d = get_yelp_full_data()
    a,b,c,d = get_yelp_polarity_data()
    a,b,c,d = get_dbpedia_data()
    a,b,c,d = get_yahoo_data()
    a,b,c,d = get_amazon_full_data()
    a,b,c,d = get_amazon_polarity_data()