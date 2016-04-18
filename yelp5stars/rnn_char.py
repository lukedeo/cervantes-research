import csv
import os
import numpy as np

from cervantes.box import WordVectorBox
from cervantes.language import OneLevelEmbedding
from cervantes.nn.models import RNNClassifier
from cervantes.box import EnglishCharBox

TRAIN_FILE = "../../yelp_review_full_csv/train.csv"
TEST_FILE = "../../yelp_review_full_csv/test.csv"

LOG_FILE = "logs/gru_char_64_1.txt"
MODEL_SPEC_FILE = "models/gru_char_64_1.spec"
MODEL_WEIGHTS_FILE = "models/gru_char_64_1.weights"

EMBEDDING_FILE = "Yelp_stars_char.pkl"

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

print "Getting data in format texts / labels"
(train_texts, train_labels) = shuffle_data(*parse_file(TRAIN_FILE))
(test_texts, test_labels) = shuffle_data(*parse_file(TEST_FILE))

print "Building language embedding"
if not os.path.isfile(EMBEDDING_FILE):

    print "Building character embedding"
    cbox = EnglishCharBox(vector_dim=100)

    # Build the language embedding with the given vector box and 2000 chars per text
    lembedding = OneLevelEmbedding(cbox, type=OneLevelEmbedding.CHAR_EMBEDDING, size=2000)
    lembedding.compute(train_texts + test_texts)
    lembedding.set_labels(train_labels + test_labels)

    print "Saving embedding"
    lembedding.save(EMBEDDING_FILE)
else:
    print "Embedding already created, loading"
    lembedding = OneLevelEmbedding.load(EMBEDDING_FILE)

# Create a recurrent neural network model and train it, the data from the computed
# embedding must be used
gru = RNNClassifier(lembedding, num_classes=5, unit='gru',
                    rnn_size=64, train_vectors=True)

gru.train(X=lembedding.data[:len(train_labels)], y=lembedding.labels[:len(train_labels)],
          model_weights_file=MODEL_WEIGHTS_FILE, model_spec_file=MODEL_SPEC_FILE)
gru.test_sequential(X=lembedding.data[len(train_labels):],
                    y=lembedding.labels[len(train_labels):])
gru.log_results(LOG_FILE, X_test=lembedding.data[len(train_labels):],
                y_test=lembedding.labels[len(train_labels):])