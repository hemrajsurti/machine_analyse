import nltk
import random
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from os.path import isfile
from io import open
nltk.data.path.append("E:\\workspace\\prj\\nltk_data")
from functools import wraps
import gc

class FetchData():
    def __init__(self, filepath):
        self.filepath = filepath
    def pickle_data_exists(self):
        return isfile(self.filepath)
    def save_data(self, data):
        data_file = open(self.filepath, "wb")
        pickle.dump(data, data_file)
        data_file.close()
        del data
    def get_data(self):
        data_file = open(self.filepath, "rb")
        data = pickle.load(data_file)
        data_file.close()
        return data
    def process(self, data):
        if not self.pickle_data_exists(self.filepath):
            self.save_pikeled_data(data, self.filepath)



class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

short_pos = open("short_reviews/positive.txt", "r", encoding='latin-1').read()
short_neg = open("short_reviews/negative.txt", "r", encoding='latin-1').read()

# move this up here
all_words = []
documents = []
#  j is adject, r is adverb, and v is verb
#allowed_word_types = ["J","R","V"]


def extra_dataset(data, data_type, allowed_word_types=["J"]):
    for p in data.split('\n'):
        documents.append((p, data_type))
        words = word_tokenize(p)
        pos = nltk.pos_tag(words)
        for w in pos:
            if w[1][0] in allowed_word_types:
                #print w
                all_words.append(w[0].lower())
    return all_words, documents

documents_file = "pickled_algos/documents.pickle"

data_set = FetchData(documents_file)
if not data_set.pickle_data_exists():
    extra_dataset(short_pos, "pos")
    extra_dataset(short_neg, "neg")
    data_set.save_data((all_words, documents))
all_words, documents = data_set.get_data()
#print all_words[:10]
word_features_file = "pickled_algos/word_features5k.pickle"

data_set = FetchData(word_features_file)
if not data_set.pickle_data_exists():
    all_words = nltk.FreqDist(all_words)
    word_features = list(all_words.keys())[:1000]
    data_set.save_data(word_features)
save_word_features = data_set.get_data()
word_features = save_word_features

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        #print w, w in words
        features[w] = (w in words)
    return features

featuresets_file = "pickled_algos/featuresets.pickle"

data_set = FetchData(featuresets_file)
if not data_set.pickle_data_exists():
    print "Processing"
    featuresets = [(find_features(rev), category) for (rev, category) in documents]
    data_set.save_data(featuresets)
featuresets = data_set.get_data()

random.shuffle(featuresets)
print(len(featuresets))
#print featuresets


testing_set = featuresets[500:]
training_set = featuresets[:500]

def train_classifier(classifier_type, classifier_obj, training_set):
    print "Processing {} Algo".format(classifier_type)
    classifier = getattr(classifier_obj, 'train')(training_set)
    perc_accuracy = nltk.classify.accuracy(classifier, testing_set)*100
    print "{} Algo accuracy percent: {}".format(classifier_type, perc_accuracy)
    return classifier


###############
save_classifier_file = "pickled_algos/originalnaivebayes5k.pickle"

data_set = FetchData(save_classifier_file)
if not data_set.pickle_data_exists():
    classifier = train_classifier('NaiveBayesClassifier',
                                  nltk.NaiveBayesClassifier, training_set)
    data_set.save_data(classifier)
    del classifier

gc.collect()

save_classifier_file = "pickled_algos/MNB_classifier5k.pickle"
data_set = FetchData(save_classifier_file)
if not data_set.pickle_data_exists():
    MNB_classifier = SklearnClassifier(MultinomialNB())
    classifier = train_classifier('MNB_classifier',
                                  MNB_classifier, training_set)
    data_set.save_data(classifier)
    del classifier


save_classifier_file = "pickled_algos/BernoulliNB_classifier5k.pickle"
data_set = FetchData(save_classifier_file)
if not data_set.pickle_data_exists():
    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    classifier = train_classifier('BernoulliNB_classifier',
                                  BernoulliNB_classifier, training_set)
    data_set.save_data(classifier)
    del classifier

save_classifier_file = "pickled_algos/LogisticRegression_classifier5k.pickle"
data_set = FetchData(save_classifier_file)
if not data_set.pickle_data_exists():
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    classifier = train_classifier('LogisticRegression_classifier',
                                  LogisticRegression_classifier, training_set)
    data_set.save_data(classifier)
    del classifier

save_classifier_file = "pickled_algos/LinearSVC_classifier5k.pickle"
data_set = FetchData(save_classifier_file)
if not data_set.pickle_data_exists():
    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    classifier = train_classifier('LinearSVC_classifier',
                                  LinearSVC_classifier, training_set)
    data_set.save_data(classifier)
    del classifier

##NuSVC_classifier = SklearnClassifier(NuSVC())
##NuSVC_classifier.train(training_set)
##print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

save_classifier_file = "pickled_algos/SGDC_classifier5k.pickle"
data_set = FetchData(save_classifier_file)
if not data_set.pickle_data_exists():
    SGDC_classifier = SklearnClassifier(SGDClassifier())
    classifier = train_classifier('SGDC_classifier',
                                  SGDC_classifier, training_set)
    data_set.save_data(classifier)
    del classifier

classifier = data_set.get_data()
MNB_classifier = data_set.get_data()
BernoulliNB_classifier = data_set.get_data()
LogisticRegression_classifier = data_set.get_data()
LinearSVC_classifier = data_set.get_data()
SGDC_classifier = data_set.get_data()
voted_classifier = VoteClassifier(classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)


def sentiment(text):
    feats = find_features(text)
    print voted_classifier.classify(feats), voted_classifier.confidence(feats)