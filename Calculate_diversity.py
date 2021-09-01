#copied from https://github.com/neural-dialogue-metrics/Distinct-N/blob/master/distinct_n/metrics.py
import nltk
import numpy as np

def distinct_n_sample_level(sample,n):
    if len(sample) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(nltk.ngrams(sample.split(), n))
    #print(len(distinct_ngrams))
    return len(distinct_ngrams) / len(sample.split())
    #return len(distinct_ngrams) / len(list(nltk.ngrams(sample.split(), n)))

def distinct_n_corpus_level(corpus,n):
    #print(sum(distinct_n_sample_level(sentence, n) for sentence in corpus) , len(corpus))
    return sum(distinct_n_sample_level(sentence, n) for sentence in corpus) / len(corpus)


def report_distinct_n(file_location, n_list):
    with open(file_location, encoding='utf-8') as f:
        corpus = f.readlines()
    for n in n_list:
        print("Distinct",n)
        print(distinct_n_corpus_level(corpus,n))

def ent_n_sample_level(sample,n):
    if len(sample) == 0:
        return 0.0  # Prevent a zero division
    tokens = nltk.word_tokenize(sample)
    n_grams = nltk.ngrams(tokens, n)
    if len(tokens) < n:
        return 0.0  # Prevent a zero division
    fdist = nltk.FreqDist(n_grams)

    f_dist_summed_count = sum(fdist.values())

    return ((-1/f_dist_summed_count) * sum([fdist[ngram]*np.log(fdist[ngram]/f_dist_summed_count) for ngram in fdist.keys()]))


def ent_n_corpus_level(corpus,n):
    #print(sum(distinct_n_sample_level(sentence, n) for sentence in corpus) , len(corpus))
    return sum(ent_n_sample_level(sentence, n) for sentence in corpus) / len(corpus)


def report_ent_n(file_location, n_list):
    with open(file_location, encoding='utf-8') as f:
        corpus = f.readlines()
    for n in n_list:
        print("Entropy",n)
        print(ent_n_corpus_level(corpus,n))

file_location = "D:\\Squad_test\\top_k_30.txt"


report_ent_n(file_location, [1,2,3,4])

report_distinct_n(file_location, [1,2,3])