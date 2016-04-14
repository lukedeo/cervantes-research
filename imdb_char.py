import os
import numpy as np

from nlpdatahandlers import ImdbDataHandler, YelpDataHandler

from cervantes.box import EnglishCharBox
from cervantes.language.embeddings import OneLevelEmbedding, TwoLevelsEmbedding
from cervantes.nn.models import RNNClassifier, LanguageClassifier

IMDB_DATA = '../deep-text/datasets/aclImdb/aclImdb'
WV_FILE = '../deep-text/embeddings/wv/glove.42B.300d.120000.txt'

if __name__ == '__main__':

    print "Getting data in format texts / labels"

    imdb = ImdbDataHandler(source=IMDB_DATA)
    (train_reviews, train_labels) = imdb.get_data(type=ImdbDataHandler.DATA_TRAIN)
    (test_reviews, test_labels) = imdb.get_data(type=ImdbDataHandler.DATA_TEST)
    train_labels = list(train_labels)
    test_labels = list(test_labels)

    print "Building language embeddings. This requires parsing text so it might " \
          "be pretty slow "

    # Compute text embeddings, containing the processed text tokens together with a vector-to-index
    # translation object (the vector box), should be pickled in order to be efficiently used with
    # different models. Hence, we can save time once we have precomputed a language embedding
    EMBEDDING_FILE = "IMDB_char_100.pkl"
    if not os.path.isfile(EMBEDDING_FILE):

        print "Building character embedding"
        cbox = EnglishCharBox(vector_dim=100)

        # Build the language embedding with the given vector box and 300 words per text
        lembedding = OneLevelEmbedding(cbox, type=OneLevelEmbedding.CHAR_EMBEDDING, size=2500)
        lembedding.compute(train_reviews + test_reviews)

        print "Saving embedding"
        lembedding.save(EMBEDDING_FILE)
    else:
        lembedding = OneLevelEmbedding.load(EMBEDDING_FILE)

    # Create a recurrent neural network model and train it, the data from the computed
    # embedding must be used
    gru = RNNClassifier(lembedding, num_classes=2, unit='gru',
                        rnn_size=64, train_vectors=True)
    gru.train(X=lembedding.data[:25000], y=train_labels, save_model_file="imdb_char_model_temp")
    gru.log_results("logs/imdb_rnn_gru.txt", X_test=lembedding.data[25000:], y_test=test_labels)
    gru.save_model("imdb_charrnn_100d_model_spec", "imdb_charrnn_100d_model_weights")
