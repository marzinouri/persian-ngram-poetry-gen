from __future__ import unicode_literals
from hazm import *
from itertools import product
from PersianG2p import Persian_g2p_converter

import random 
import numpy as np
import nltk
import pandas as pd
import codecs
import tqdm
import math
import nltk


mesra_collection = [x.strip().split() for x in tqdm.tqdm(codecs.open('ferdousi_norm.txt','rU','utf-8').readlines())]


# remove empty lines
def remove_items(test_list, item):
    res = [i for i in test_list if i != item]
    return res
mesra_collection = remove_items(mesra_collection, [])



normalizer = Normalizer()
mesra_normalized = [[normalizer.normalize(y) for y in x] for x in tqdm.tqdm(mesra_collection)]

mesra_sentences = [sent_tokenize(' '.join(x)) for x in tqdm.tqdm(mesra_normalized)]

mesra_tokens = [[word_tokenize(sent) for sent in sents][0] for sents in tqdm.tqdm(mesra_sentences)]

mesra_tokens = [' '.join(x) for x in mesra_tokens]

PersianG2Pconverter = Persian_g2p_converter()

def get_phonemes_list(tokens):
    phonemes = []
    for token in tokens:
        phoneme = PersianG2Pconverter.transliterate(token, tidy = False, secret=True)
        phonemes.append(phoneme)
    return phonemes

def extract_qafiye_heja(qafiye):
    qafiye_phoneme = PersianG2Pconverter.transliterate(qafiye, tidy = False, secret=True)
    vowels = ['a','A','i','e','o','u']
    for i, c in enumerate(reversed(qafiye_phoneme)):
        qafiye_heja = ""
        l = len(qafiye_phoneme)
        if c in vowels:   
            qafiye_heja = qafiye_phoneme[l-i-2:]
            break
    return qafiye_heja

def choose_hamqafiye(qafiye, tokens):
    phonemes = get_phonemes_list(tokens)
    ham_qafiye_tokens = []
    qafiye_heja = extract_qafiye_heja(qafiye)
    l = len(qafiye_heja)
    for index, phoneme in enumerate(phonemes):
        if phoneme[-l:] == qafiye_heja and tokens[index] not in ham_qafiye_tokens and tokens[index]!=qafiye:
            ham_qafiye_tokens.append(tokens[index])
    hamqafiye = random.choice(ham_qafiye_tokens)
    return hamqafiye


# reversed

# Modified version of 
# https://github.com/joshualoehr/ngram-language-model/blob/master/language_model.py


class ReversedLanguageModel(object):
    """An n-gram language model trained on a given corpus.
    
    For a given n and given training corpus, constructs an n-gram language
    model for the corpus by:
    1. preprocessing the corpus (adding SOS/EOS/UNK tokens)
    2. calculating (smoothed) probabilities for each n-gram
    Also contains methods for calculating the perplexity of the model
    against another corpus, and for generating sentences.
    Args:
        train_data (list of str): list of sentences comprising the training corpus.
        n (int): the order of language model to build (i.e. 1 for unigram, 2 for bigram, etc.).
        laplace (int): lambda multiplier to use for laplace smoothing (default 1 for add-1 smoothing).
    """

    SOS = "<s>"
    EOS = "</s>"
    UNK = "<UNK>"
    
    def __init__(self, train_data, n, laplace=1):
        self.n = n
        self.vocab = dict()
        self.laplace = laplace
        self.tokens = self.preprocess(train_data, n)
        self.vocab  = nltk.FreqDist(self.tokens)
        self.model  = self._create_model()
        self.masks  = list(reversed(list(product((0,1), repeat=n))))

    def _smooth(self):
        """Apply Laplace smoothing to n-gram frequency distribution.
        
        Here, n_grams refers to the n-grams of the tokens in the training corpus,
        while m_grams refers to the first (n-1) tokens of each n-gram.
        Returns:
            dict: Mapping of each n-gram (tuple of str) to its Laplace-smoothed 
            probability (float).
        """
        vocab_size = len(self.vocab)

        n_grams = nltk.ngrams(self.tokens, self.n)
        n_vocab = nltk.FreqDist(n_grams)

        m_grams = nltk.ngrams(self.tokens, self.n-1)
        m_vocab = nltk.FreqDist(m_grams)

        def smoothed_count(n_gram, n_count):
            m_gram = n_gram[:-1]
            m_count = m_vocab[m_gram]
            return (n_count + self.laplace) / (m_count + self.laplace * vocab_size)

        return { n_gram: smoothed_count(n_gram, count) for n_gram, count in n_vocab.items() }

    def _create_model(self):
        """Create a probability distribution for the vocabulary of the training corpus.
        
        If building a unigram model, the probabilities are simple relative frequencies
        of each token with the entire corpus.
        Otherwise, the probabilities are Laplace-smoothed relative frequencies.
        Returns:
            A dict mapping each n-gram (tuple of str) to its probability (float).
        """
        if self.n == 1:
            num_tokens = len(self.tokens)
            return { (unigram,): count / num_tokens for unigram, count in self.vocab.items() }
        else:
            return self._smooth()

    def _convert_oov(self, ngram):
        """Convert, if necessary, a given n-gram to one which is known by the model.
        Starting with the unmodified ngram, check each possible permutation of the n-gram
        with each index of the n-gram containing either the original token or <UNK>. Stop
        when the model contains an entry for that permutation.
        This is achieved by creating a 'bitmask' for the n-gram tuple, and swapping out
        each flagged token for <UNK>. Thus, in the worst case, this function checks 2^n
        possible n-grams before returning.
        Returns:
            The n-gram with <UNK> tokens in certain positions such that the model
            contains an entry for it.
        """
        mask = lambda ngram, bitmask: tuple((token if flag == 1 else "<UNK>" for token,flag in zip(ngram, bitmask)))

        ngram = (ngram,) if type(ngram) is str else ngram
        for possible_known in [mask(ngram, bitmask) for bitmask in self.masks]:
            if possible_known in self.model:
                return possible_known

    def perplexity(self, test_data):
        """Calculate the perplexity of the model against a given test corpus.
        
        Args:
            test_data (list of str): sentences comprising the training corpus.
        Returns:
            The perplexity of the model as a float.
        
        """
        test_tokens = self.preprocess(test_data, self.n)
        test_ngrams = nltk.ngrams(test_tokens, self.n)
        N = len(test_tokens)

        known_ngrams  = [self._convert_oov(ngram) for ngram in test_ngrams]
        probabilities = [self.model[ngram] for ngram in known_ngrams]
        
        for x,y in zip(known_ngrams, probabilities):
            print(x,y)
        
        return math.exp((-1/N) * sum(map(math.log, probabilities)))

    def _best_candidate(self, prev, without=[]):
        
        blacklist  = [ReversedLanguageModel.UNK] + without

        if len(prev) < self.n:
            prev = [ReversedLanguageModel.SOS]*(self.n-1)

        candidates = list(((ngram[-1],prob) for ngram,prob in self.model.items() if ngram[:-1]==tuple(prev)))

        probs = [y for x,y in candidates]
        probs = probs/np.sum(probs)
        words = [x for x,y in candidates]

        idx = np.random.choice(len(words), 1, replace=False, p=probs)[0]
        
        while words[idx] in blacklist:
            idx = np.random.choice(len(words), 1, replace=False, p=probs)[0]
        
        return (words[idx], probs[idx])
    
    
    def generate_sentence(self, first_mesra, use_qafiye):
        target_len = len(first_mesra.split())

        qafiye_word =first_mesra.split()[-1]
        if use_qafiye==1:
            qafiye = choose_hamqafiye(qafiye_word, self.tokens[:1000])
        
        eos = ' '.join([ReversedLanguageModel.EOS] * (self.n-1)) if self.n > 1 else ReversedLanguageModel.EOS
        if use_qafiye==1:
            first_mesra_with_qafiye = qafiye + ' ' + first_mesra
            first_mesra_tokens = first_mesra_with_qafiye.split()
        else:
            first_mesra_tokens = first_mesra.split()
            
        first_mesra_part = first_mesra_tokens[:(self.n-1)]
        first_mesra_part.reverse()
        sent, prob = (first_mesra_part, 1)
        
        if use_qafiye==1:
            min_len = target_len + len(sent) - 2
            max_len = target_len + len(sent) - 1
        else:
            min_len = target_len + len(sent) - 1
            max_len = target_len + len(sent) 
            
        while sent[-1] != ReversedLanguageModel.SOS:
            prev = () if self.n == 1 else tuple(sent[-(self.n-1):])
            blacklist = sent + ([ReversedLanguageModel.EOS,ReversedLanguageModel.SOS] if len(sent) < min_len else [])
            next_token, next_prob = self._best_candidate(prev, without=blacklist)
            sent.append(next_token)
            prob *= next_prob

            if len(sent) >= max_len:
                sent.append(ReversedLanguageModel.SOS)
        remaining_part = sent[(self.n-1):-1]
        remaining_part.reverse()
        if use_qafiye == 1:
            return (first_mesra+' *** '+' '.join(remaining_part)+' '+qafiye, -1/math.log(prob))
        else:
            return (first_mesra+' *** '+' '.join(remaining_part), -1/math.log(prob))
        

    def add_sentence_tokens(self, sentences, n):
        """Wrap each sentence in SOS and EOS tokens.
        For n >= 2, n-1 SOS tokens are added, otherwise only one is added.
        Args:
            sentences (list of str): the sentences to wrap.
            n (int): order of the n-gram model which will use these sentences.
        Returns:
            List of sentences with SOS and EOS tokens wrapped around them.
        """
        sos = ' '.join([ReversedLanguageModel.SOS] * (n-1)) if n > 1 else ReversedLanguageModel.SOS
        return [sos+' '+s+' '+ReversedLanguageModel.EOS for s in sentences] 

    def replace_singletons(self, tokens):
        """Replace tokens which appear only once in the corpus with <UNK>.

        Args:
            tokens (list of str): the tokens comprising the corpus.
        Returns:
            The same list of tokens with each singleton replaced by <UNK>.

        """
        if len(self.vocab) == 0:
            self.vocab = nltk.FreqDist(tokens)
        return [token if self.vocab[token] > 1 else ReversedLanguageModel.UNK for token in tokens]

    def preprocess(self, sentences, n):
        """Add SOS/EOS/UNK tokens to given sentences and tokenize.
        Args:
            sentences (list of str): the sentences to preprocess.
            n (int): order of the n-gram model which will use these sentences.
        Returns:
            The preprocessed sentences, tokenized by words.
        """
        sentences = self.add_sentence_tokens(sentences, n)
        tokens = ' '.join(sentences).split()
        tokens = self.replace_singletons(tokens)
        return tokens 
    
n = 5
rlm = ReversedLanguageModel(mesra_tokens, n)
first_mesra = ["بسی رنج بردم در این سال سی"]
use_qafiye = 1
beyt = rlm.generate_sentence(first_mesra[0], use_qafiye)

print(beyt[0])