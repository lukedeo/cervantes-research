import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle

WORDS_FILE = "AG_News_words.pkl"

text_words = pickle.load(open(WORDS_FILE))
text_lens = [len(x) for x in text_words]

print "Percentile 90: " + str(np.percentile(text_lens, 90))
print "Percentile 95: " + str(np.percentile(text_lens, 95))
print "Percentile 99: " + str(np.percentile(text_lens, 99))
print "Percentile 99.5: " + str(np.percentile(text_lens, 99.5))
print "Percentile 99.9: " + str(np.percentile(text_lens, 99.9))


plt.hist(text_lens, 20)
plt.show()