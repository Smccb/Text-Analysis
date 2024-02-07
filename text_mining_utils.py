import re, pandas as pd, numpy as np, matplotlib.pyplot as plt;
import nltk;
from collections import Counter;
import warnings;
warnings.filterwarnings('ignore');
from sklearn.feature_extraction.text import CountVectorizer;
from sklearn.metrics import classification_report;
from sklearn.model_selection import cross_val_predict;


#############################################
##### visualise POS of docs by category #####

"""
takes in a list of tagged documents and a POS(as a string);
returns the normalised count of a POS for each tagged document;
"""
def normalisePOSCounts(tagged_docs, pos):
	counts = []
	for doc in tagged_docs:
		count = 0
		for pair in doc:
			if pair[1] == pos:
				count += 1
		counts.append(count)
	lengths = [len(doc) for doc in tagged_docs]
	return [count/length for count, length in zip(counts, lengths)]

"""
takes in a list of documents, a POS(as a string), and a list of categories/labels;
it tags the documents and calls the above function;
it then plots the normalised frequency of the POS across all labels;
"""
def plotPOSFreq(docs, pos, labels):
	tagged_docs = [nltk.pos_tag(nltk.word_tokenize(doc)) for doc in docs]
	normalised_counts = normalisePOSCounts(tagged_docs, pos)
	plt.bar(np.arange(len(docs)), normalised_counts, align='center')
	plt.xticks(np.arange(len(docs)), labels, rotation=40)
	plt.xlabel('Label (Category)')
	plt.ylabel(pos + ' frequency')
	plt.title('Frequency distribution of ' + pos)


#########################################################
########## vectorisation functions ######################

""""
NOTE: all vectorisers from sklearn discard punctuation, which may not be appropriate.
So, I have specified a regex to deal with this situation.
"""
token_regex = r"\w+(?:'\w+)?|[^\w\s]";

"""
takes in a list of documents, applies the CountVectoriser from sklearn
using the following params by default: decode_error='replace', strip_accents=None, 
lowercase=False, ngram_range=(1, 1); then it builds and returns a data frame;
"""
def build_count_matrix(docs, decode_error='replace', strip_accents=None, lowercase=False, token_pattern=token_regex, ngram_range=(1, 1)):    
    vectorizer = CountVectorizer(decode_error=decode_error, strip_accents=strip_accents, lowercase=lowercase, token_pattern=token_pattern, ngram_range=ngram_range)
    X = vectorizer.fit_transform(docs)
    terms = list(vectorizer.get_feature_names_out())
    count_matrix = pd.DataFrame(X.toarray(), columns=terms)
    return count_matrix.fillna(0)
#########################################################
############# validation functions ######################

"""function to train and x-validate across acc, rec, prec; 
and get the classification report"""
def printClassifReport(clf, X, y, folds=5):
    predictions = cross_val_predict(clf, X, y, cv=folds)
    print(classification_report(y, predictions))