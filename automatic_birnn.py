import os
import numpy as np

from cervantes.box import WordVectorBox
from cervantes.language import OneLevelEmbedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datasets
from cervantes.nn.models import BiRNNClassifier

def gogogo_wv(train_texts, train_labels, test_texts, test_labels,
              embedding_file, embedding_size, wv_file,
              model_weights_file, model_spec_file, log_file,
              num_classes, keras_params):

    if not os.path.isfile(embedding_file):
        # We need a file with precomputed wordvectors
        print 'Building global word vectors from {}'.format(wv_file)

        gbox = WordVectorBox(wv_file)
        gbox.build(zero_token=True, normalize_variance=False, normalize_norm=True)

        # Build the language embedding with the given vector box and 100 words per text
        print "Computing embedding"
        lembedding = OneLevelEmbedding(gbox, size=embedding_size)
        lembedding.compute(train_texts + test_texts)
        lembedding.set_labels(train_labels + test_labels)
        lembedding.save(embedding_file)
    else:
        lembedding = OneLevelEmbedding.load(embedding_file)

    #  MODEL DEPENDENT
    clf = BiRNNClassifier(lembedding, num_classes=num_classes, unit='gru',
                        rnn_size=64, train_vectors=True)
    # ! MODEL DEPENDENT
    clf.train(X=lembedding.data[:len(train_labels)], y=lembedding.labels[:len(train_labels)],
              model_weights_file=model_weights_file, model_spec_file=model_spec_file, **keras_params)
    clf.test(X=lembedding.data[len(train_labels):],
                        y=lembedding.labels[len(train_labels):])
    clf.log_results(log_file, X_test=lembedding.data[len(train_labels):],
                    y_test=lembedding.labels[len(train_labels):])


WV_FILE = '../deep-text/embeddings/wv/glove.42B.300d.120000.txt'
RESULTS_DIR = "./experiments/bigru_64_results/"
FIT_PARAMS = {
    "batch_size": 32,
    "nb_epoch": 100,
    "verbose": 2,
    "validation_split": 0.15,
    "callbacks": [EarlyStopping(verbose=True, patience=2, monitor='val_acc')]
}

print("==" * 40)
print("AG_NEWS")
print("==" * 40)
train_texts, train_labels, test_texts, test_labels = datasets.get_ag_news_data()
gogogo_wv(train_texts=train_texts, train_labels=train_labels,
          test_texts=test_texts, test_labels=test_labels,
          embedding_file="AGNews_WVs.pkl", embedding_size=100, wv_file=WV_FILE,
          model_weights_file=RESULTS_DIR + "ag_news.weights",
          model_spec_file=RESULTS_DIR + "ag_news.spec",
          log_file=RESULTS_DIR + "ag_news.log",
          num_classes=4,
          keras_params=FIT_PARAMS)

#datasets.get_sogou_data()

print("==" * 40)
print("DBPedia")
print("==" * 40)
train_texts, train_labels, test_texts, test_labels = datasets.get_dbpedia_data()
FIT_PARAMS = {
    "batch_size": 32,
    "nb_epoch": 100,
    "verbose": 2,
    "validation_split": 0.15,
    "callbacks": [EarlyStopping(verbose=True, patience=2, monitor='val_acc')]
}
gogogo_wv(train_texts=train_texts, train_labels=train_labels,
          test_texts=test_texts, test_labels=test_labels,
          embedding_file="DBPedia_WVs.pkl", embedding_size=140, wv_file=WV_FILE,
          model_weights_file=RESULTS_DIR + "dbpedia.weights",
          model_spec_file=RESULTS_DIR + "dbpedia.spec",
          log_file=RESULTS_DIR + "dbpedia.log",
          num_classes=14,
          keras_params=FIT_PARAMS)

print("==" * 40)
print("Yelp Polarity")
print("==" * 40)
train_texts, train_labels, test_texts, test_labels = datasets.get_yelp_polarity_data()
FIT_PARAMS = {
    "batch_size": 32,
    "nb_epoch": 100,
    "verbose": 2,
    "validation_split": 0.15,
    "callbacks": [EarlyStopping(verbose=True, patience=2, monitor='val_acc')]
}
gogogo_wv(train_texts=train_texts, train_labels=train_labels,
          test_texts=test_texts, test_labels=test_labels,
          embedding_file="YelpPolarity_WVs.pkl", embedding_size=500, wv_file=WV_FILE,
          model_weights_file=RESULTS_DIR + "yelp_polarity.weights",
          model_spec_file=RESULTS_DIR + "yelp_polarity.spec",
          log_file=RESULTS_DIR + "yelp_polarity.log",
          num_classes=2,
          keras_params=FIT_PARAMS)

print("==" * 40)
print("Yelp Full Data")
print("==" * 40)
train_texts, train_labels, test_texts, test_labels = datasets.get_yelp_full_data()
FIT_PARAMS = {
    "batch_size": 32,
    "nb_epoch": 100,
    "verbose": 2,
    "validation_split": 0.15,
    "callbacks": [EarlyStopping(verbose=True, patience=2, monitor='val_acc')]
}
gogogo_wv(train_texts=train_texts, train_labels=train_labels,
          test_texts=test_texts, test_labels=test_labels,
          embedding_file="YelpFull_WVs.pkl", embedding_size=500, wv_file=WV_FILE,
          model_weights_file=RESULTS_DIR + "yelp_full.weights",
          model_spec_file=RESULTS_DIR + "yelp_full.spec",
          log_file=RESULTS_DIR + "yelp_full.log",
          num_classes=5,
          keras_params=FIT_PARAMS)

print("==" * 40)
print("Yahoo")
print("==" * 40)
train_texts, train_labels, test_texts, test_labels = datasets.get_yahoo_data()
FIT_PARAMS = {
    "batch_size": 32,
    "nb_epoch": 100,
    "verbose": 2,
    "validation_split": 0.15,
    "callbacks": [EarlyStopping(verbose=True, patience=2, monitor='val_acc')]
}
gogogo_wv(train_texts=train_texts, train_labels=train_labels,
          test_texts=test_texts, test_labels=test_labels,
          embedding_file="Yahoo_WVs.pkl", embedding_size=400, wv_file=WV_FILE,
          model_weights_file=RESULTS_DIR + "yahoo.weights",
          model_spec_file=RESULTS_DIR + "yahoo.spec",
          log_file=RESULTS_DIR + "yahoo.log",
          num_classes=10,
          keras_params=FIT_PARAMS)

print("==" * 40)
print("Amazon polarity")
print("==" * 40)
train_texts, train_labels, test_texts, test_labels = datasets.get_amazon_polarity_data()
FIT_PARAMS = {
    "batch_size": 32,
    "nb_epoch": 100,
    "verbose": 2,
    "validation_split": 0.15,
    "callbacks": [EarlyStopping(verbose=True, patience=2, monitor='val_acc')]
}
gogogo_wv(train_texts=train_texts, train_labels=train_labels,
          test_texts=test_texts, test_labels=test_labels,
          embedding_file="AmazonPolarity_WVs.pkl", embedding_size=200, wv_file=WV_FILE,
          model_weights_file=RESULTS_DIR + "amazon_polarity.weights",
          model_spec_file=RESULTS_DIR + "amazon_polarity.spec",
          log_file=RESULTS_DIR + "amazon_polarity.log",
          num_classes=2,
          keras_params=FIT_PARAMS)

print("==" * 40)
print("Amazon Full")
print("==" * 40)
train_texts, train_labels, test_texts, test_labels = datasets.get_amazon_full_data()
FIT_PARAMS = {
    "batch_size": 32,
    "nb_epoch": 100,
    "verbose": 2,
    "validation_split": 0.15,
    "callbacks": [EarlyStopping(verbose=True, patience=2, monitor='val_acc')]
}
gogogo_wv(train_texts=train_texts, train_labels=train_labels,
          test_texts=test_texts, test_labels=test_labels,
          embedding_file="AmazonFull_WVs.pkl", embedding_size=200, wv_file=WV_FILE,
          model_weights_file=RESULTS_DIR + "amazon_full.weights",
          model_spec_file=RESULTS_DIR + "amazon_full.spec",
          log_file=RESULTS_DIR + "amazon_full.log",
          num_classes=5,
          keras_params=FIT_PARAMS)
