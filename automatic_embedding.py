import os

print "Importing embeddings" 
from cervantes.box import EnglishCharBox
from cervantes.language import TwoLevelsEmbedding
import datasets
from joblib import Parallel, delayed

N_JOBS = 39
CHARACTERS_PER_WORD = 15

def _build_embeddings(train_texts, train_labels, test_texts, test_labels, 
                     embedding_file, embedding_size, vector_dim):

    print "Building character embedding"
    cbox = EnglishCharBox(vector_dim=vector_dim)

    # Build the language embedding with the given vector box and 2000 chars
    # per text
    size_level1, size_level2 = (
        CHARACTERS_PER_WORD, embedding_size / CHARACTERS_PER_WORD)

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
    del lembedding



def construct(dataset, embedding_size):
    assert dataset in {'ag_news', 'sogou', 'dbpedia', 'yelp_polarity', 
                       'yelp_full', 'yahoo', 'amazon_polarity', 'amazon_full'}
                       
    _build_embeddings(*getattr(datasets, 'get_%s_data' % dataset),
                      embedding_file='embeddings/%s_len_%i_chars.pkl' % (
                      dataset, embedding_size), embedding_size=embedding_size,
                      vector_dim=50)


if __name__ == '__main__':
    
    datasets = ['ag_news', 'sogou', 'dbpedia', 'yelp_polarity', 'yelp_full', 
                'yahoo', 'amazon_polarity', 'amazon_full']
    sizes = [800, 2000, 800, 2000, 2000, 2000, 1000, 1000]


    Parallel(n_jobs=N_JOBS)(delayed(construct)(ds, sz) for (ds, sz) in zip(datasets, sizes))






