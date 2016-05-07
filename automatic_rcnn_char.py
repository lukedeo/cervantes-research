import os
import numpy as np

print "Importing embeddings" 
from cervantes.box import WordVectorBox, EnglishCharBox
from cervantes.language import OneLevelEmbedding, TwoLevelsEmbedding
from keras.callbacks import EarlyStopping, ModelCheckpoint


CHARACTERS_PER_WORD = 15

def gogogo_char(train_texts, train_labels, test_texts, test_labels,
              embedding_file, embedding_size,
              model_weights_file, model_spec_file, log_file,
              num_classes, keras_params):

    print "Building language embedding"
    if not os.path.isfile(embedding_file):

        print "Building character embedding"
        cbox = EnglishCharBox(vector_dim=50)

        # Build the language embedding with the given vector box and 2000 chars
        # per text
        # size_level1, size_level2 = CHARACTERS_PER_WORD, WORDS_PER_DOCUMENT
        size_level1, size_level2 = (CHARACTERS_PER_WORD, embedding_size / CHARACTERS_PER_WORD)

        lembedding = TwoLevelsEmbedding(
            vector_box=cbox,
            type=TwoLevelsEmbedding.CHAR_WORD_EMBEDDING,
            size_level1=size_level1,
            size_level2=size_level2
        )

        lembedding.compute(train_texts + test_texts)
        lembedding.set_labels(train_labels + test_labels)

        print "Saving embedding"
        lembedding.save(embedding_file)
    else:
        print "Embedding already created, loading"
        lembedding = TwoLevelsEmbedding.load(embedding_file)

    #  MODEL DEPENDENT
    clf = RCNNClassifier(lembedding, num_classes=num_classes, optimizer='adam')
    # ! MODEL DEPENDENT
    clf.train(X=lembedding.data[:len(train_labels)], y=lembedding.labels[:len(train_labels)],
              model_weights_file=model_weights_file, model_spec_file=model_spec_file, **keras_params)
    
    clf.test(X=lembedding.data[len(train_labels):],
             y=lembedding.labels[len(train_labels):])
    
    clf.log_results(log_file, 
                    X_test=lembedding.data[len(train_labels):],
                    y_test=lembedding.labels[len(train_labels):])


import datasets

print "Importing models"
from cervantes.nn.models import RCNNClassifier, BasicCNNClassifier


RESULTS_DIR = "./experiments/char_rcnn_gru_results/"

# FIT_PARAMS = {
#     "batch_size": 512,
#     "nb_epoch": 100,
#     "verbose": 2,
#     "validation_split": 0.15,
#     "callbacks": [EarlyStopping(verbose=True, patience=5, monitor='val_acc')]
# }

# print("==" * 40)
# print("AG_NEWS")
# print("==" * 40)
# train_texts, train_labels, test_texts, test_labels = datasets.get_ag_news_data()
# n = min(len(train_labels), 220000)
# train_texts, train_labels = train_texts[:n], train_labels[:n]
# gogogo_char(train_texts=train_texts, train_labels=train_labels,
#           test_texts=test_texts, test_labels=test_labels,
#           embedding_file="AGNews_chars.pkl", embedding_size=800,
#           model_weights_file=RESULTS_DIR + "ag_news.weights",
#           model_spec_file=RESULTS_DIR + "ag_news.spec",
#           log_file=RESULTS_DIR + "ag_news.log",
#           num_classes=4,
#           keras_params=FIT_PARAMS)

print("==" * 40)
print("Sogou")
print("==" * 40)
train_texts, train_labels, test_texts, test_labels = datasets.get_sogou_data()
n = min(len(train_labels), 220000)
train_texts, train_labels = train_texts[:n], train_labels[:n]
FIT_PARAMS = {
    "batch_size": 512,
    "nb_epoch": 100,
    "verbose": 2,
    "validation_split": 0.15,
    "callbacks": [EarlyStopping(verbose=True, patience=5, monitor='val_acc')]
}
gogogo_char(train_texts=train_texts, train_labels=train_labels,
          test_texts=test_texts, test_labels=test_labels,
          embedding_file="Sogou_chars.pkl", embedding_size=6000,
          model_weights_file=RESULTS_DIR + "sogou.weights",
          model_spec_file=RESULTS_DIR + "sogou.spec",
          log_file=RESULTS_DIR + "sogou.log",
          num_classes=5,
          keras_params=FIT_PARAMS)

print("==" * 40)
print("DBPedia")
print("==" * 40)
train_texts, train_labels, test_texts, test_labels = datasets.get_dbpedia_data()
n = min(len(train_labels), 220000)
train_texts, train_labels = train_texts[:n], train_labels[:n]
FIT_PARAMS = {
    "batch_size": 512,
    "nb_epoch": 100,
    "verbose": 2,
    "validation_split": 0.15,
    "callbacks": [EarlyStopping(verbose=True, patience=5, monitor='val_acc')]
}
gogogo_char(train_texts=train_texts, train_labels=train_labels,
          test_texts=test_texts, test_labels=test_labels,
          embedding_file="DBPedia_chars.pkl", embedding_size=800,
          model_weights_file=RESULTS_DIR + "dbpedia.weights",
          model_spec_file=RESULTS_DIR + "dbpedia.spec",
          log_file=RESULTS_DIR + "dbpedia.log",
          num_classes=14,
          keras_params=FIT_PARAMS)

print("==" * 40)
print("Yelp Polarity")
print("==" * 40)
train_texts, train_labels, test_texts, test_labels = datasets.get_yelp_polarity_data()
n = min(len(train_labels), 220000)
train_texts, train_labels = train_texts[:n], train_labels[:n]
FIT_PARAMS = {
    "batch_size": 512,
    "nb_epoch": 100,
    "verbose": 2,
    "validation_split": 0.15,
    "callbacks": [EarlyStopping(verbose=True, patience=5, monitor='val_acc')]
}
gogogo_char(train_texts=train_texts, train_labels=train_labels,
          test_texts=test_texts, test_labels=test_labels,
          embedding_file="YelpPolarity_chars.pkl", embedding_size=2000,
          model_weights_file=RESULTS_DIR + "yelp_polarity.weights",
          model_spec_file=RESULTS_DIR + "yelp_polarity.spec",
          log_file=RESULTS_DIR + "yelp_polarity.log",
          num_classes=2,
          keras_params=FIT_PARAMS)

print("==" * 40)
print("Yelp Full Data")
print("==" * 40)
train_texts, train_labels, test_texts, test_labels = datasets.get_yelp_full_data()
n = min(len(train_labels), 220000)
train_texts, train_labels = train_texts[:n], train_labels[:n]
FIT_PARAMS = {
    "batch_size": 512,
    "nb_epoch": 100,
    "verbose": 2,
    "validation_split": 0.15,
    "callbacks": [EarlyStopping(verbose=True, patience=5, monitor='val_acc')]
}
gogogo_char(train_texts=train_texts, train_labels=train_labels,
          test_texts=test_texts, test_labels=test_labels,
          embedding_file="YelpFull_chars.pkl", embedding_size=2000,
          model_weights_file=RESULTS_DIR + "yelp_full.weights",
          model_spec_file=RESULTS_DIR + "yelp_full.spec",
          log_file=RESULTS_DIR + "yelp_full.log",
          num_classes=5,
          keras_params=FIT_PARAMS)

print("==" * 40)
print("Yahoo")
print("==" * 40)
train_texts, train_labels, test_texts, test_labels = datasets.get_yahoo_data()
n = min(len(train_labels), 220000)
train_texts, train_labels = train_texts[:n], train_labels[:n]
FIT_PARAMS = {
    "batch_size": 512,
    "nb_epoch": 100,
    "verbose": 2,
    "validation_split": 0.15,
    "callbacks": [EarlyStopping(verbose=True, patience=5, monitor='val_acc')]
}
gogogo_char(train_texts=train_texts, train_labels=train_labels,
          test_texts=test_texts, test_labels=test_labels,
          embedding_file="Yahoo_chars.pkl", embedding_size=2000,
          model_weights_file=RESULTS_DIR + "yahoo.weights",
          model_spec_file=RESULTS_DIR + "yahoo.spec",
          log_file=RESULTS_DIR + "yahoo.log",
          num_classes=10,
          keras_params=FIT_PARAMS)

print("==" * 40)
print("Amazon polarity")
print("==" * 40)
train_texts, train_labels, test_texts, test_labels = datasets.get_amazon_polarity_data()
n = min(len(train_labels), 220000)
train_texts, train_labels = train_texts[:n], train_labels[:n]
FIT_PARAMS = {
    "batch_size": 512,
    "nb_epoch": 100,
    "verbose": 2,
    "validation_split": 0.15,
    "callbacks": [EarlyStopping(verbose=True, patience=5, monitor='val_acc')]
}
gogogo_char(train_texts=train_texts, train_labels=train_labels,
          test_texts=test_texts, test_labels=test_labels,
          embedding_file="AmazonPolarity_chars.pkl", embedding_size=1000,
          model_weights_file=RESULTS_DIR + "amazon_polarity.weights",
          model_spec_file=RESULTS_DIR + "amazon_polarity.spec",
          log_file=RESULTS_DIR + "amazon_polarity.log",
          num_classes=2,
          keras_params=FIT_PARAMS)

print("==" * 40)
print("Amazon Full")
print("==" * 40)
train_texts, train_labels, test_texts, test_labels = datasets.get_amazon_full_data()
n = min(len(train_labels), 220000)
train_texts, train_labels = train_texts[:n], train_labels[:n]
FIT_PARAMS = {
    "batch_size": 512,
    "nb_epoch": 100,
    "verbose": 2,
    "validation_split": 0.15,
    "callbacks": [EarlyStopping(verbose=True, patience=5, monitor='val_acc')]
}
gogogo_char(train_texts=train_texts, train_labels=train_labels,
          test_texts=test_texts, test_labels=test_labels,
          embedding_file="AmazonFull_chars.pkl", embedding_size=1000,
          model_weights_file=RESULTS_DIR + "amazon_full.weights",
          model_spec_file=RESULTS_DIR + "amazon_full.spec",
          log_file=RESULTS_DIR + "amazon_full.log",
          num_classes=5,
          keras_params=FIT_PARAMS)
