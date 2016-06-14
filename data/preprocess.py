# Modification of https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/rnn/translate/data_utils.py
#
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import gzip
import time
import tarfile
import json
from tqdm import *
from glob import glob
from collections import defaultdict
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

from tensorflow.python.platform import gfile

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"(^| )\d+")

UNK_ID = 1

tokenizer = RegexpTokenizer(r'@?\w+')
cachedStopWords = stopwords.words("english")


def basic_tokenizer(sentence):
    words = tokenizer.tokenize(sentence)
    return [w for w in words if w not in stopwords.words("english")]

def create_vobabulary(vocabulary_path, context, max_vocabulary_size = 30000,
                              tokenizer=None, normalize_digits=True):
    if not gfile.Exists(vocabulary_path):
        t0 = time.time()
        print ("[*] Creating vocabulary %s" % (vocabulary_path))
        texts = [word for word in context.lower().split() if word not in cachedStopWords]
        dictionary = corpora.Dictionary([texts], prune_at=max_vocabulary_size)
        print ("[*] Filtering extremes")
        dictionary.filter_extremes(no_above = 0.8)

        print("[*] Tokenize : %.4fs" % (t0 - time.time()))
        dictionary.save(vocabulary_path)

def initialize_vocabulary(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        vocab = corpora.Dictionary.load(vocabulary_path)
        return vocab.token2id, vocab.token2id.keys()
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def max_vocab_size(token):
    threshold = 30000
    if token <= threshold:
        return token
    else:
        return UNK_ID

def sentence_to_token_ids(sentence, vocabulary,
                              tokenizer=None, normalize_digits=True):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)

    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    else:
        return [max_vocab_size(vocabulary.get(re.sub(_DIGIT_RE, " ", w.lower()), UNK_ID)) for w in words]

def data_to_token_ids(vocab, vocab_size, tokenizer = None, normalize_digits = True):
    print("[*] Data 2 Token ids")
    print("[*] Tokenizing train.context")
    vocab_size = str(vocab_size)
    with gfile.GFile('train.context', mode="r") as trainFile:
        trainResults = []
        for line in trainFile:
            trainIds = sentence_to_token_ids(line, vocab, tokenizer, normalize_digits)
            trainResults.append(" ".join([str(tok) for tok in trainIds]) + '\n')
    with gfile.GFile('train_token.context'+vocab_size, mode="w") as tokenFile:
        tokenFile.writelines(trainResults)
        del trainResults[:]

    print("[*] Tokenizing train.opinion")
    with gfile.GFile('train.opinion', mode="r") as trainFile:
        trainResults = []
        counter = 0
        for line in trainFile:
            if counter % 6 != 0:
                trainIds = sentence_to_token_ids(line, vocab, tokenizer, normalize_digits)
                trainResults.append(" ".join([str(tok) for tok in trainIds]) + '\n')
            else:
                trainResults.append(line) # Ans.
            counter += 1
    with gfile.GFile('train_token.opinion'+vocab_size, mode="w") as tokenFile:
        tokenFile.writelines(trainResults)
        del trainResults[:]

    print("[*] Tokenizing test.context")
    with gfile.GFile('test.context', mode="r") as testFile:
        testResults = []
        for line in testFile:
            testIds = sentence_to_token_ids(line, vocab, tokenizer, normalize_digits)
            testResults.append(" ".join([str(tok) for tok in testIds]) + '\n' )
    with gfile.GFile('test_token.context'+vocab_size, mode="w") as tokenFile:
        tokenFile.writelines(testResults)
        del testResults[:]

    print("[*] Tokenizing test.opinion")
    with gfile.GFile('test.opinion', mode="r") as testFile:
        testResults = []
        counter = 0
        for line in testFile:
            if counter % 6 != 0:
                testIds = sentence_to_token_ids(line, vocab, tokenizer, normalize_digits)
                testResults.append(" ".join([str(tok) for tok in testIds]) + '\n')
            counter += 1
    with gfile.GFile('test_token.opinion'+vocab_size, mode="w") as tokenFile:
        tokenFile.writelines(testResults)
        del testResults[:]

    return

def questions_to_token_ids(vocab_fname, vocab_size):
    vocab, _ = initialize_vocabulary(vocab_fname)
    data_to_token_ids(vocab, vocab_size)

def prepare_data(dataset_name, vocab_size):
    vocab_fname = os.path.join('%s.vocab%s' % (dataset_name, vocab_size))
    if not os.path.exists(dataset_name):
        raise ValueError("Data file %s not found.", dataset_name)
    # collect.data
    context = gfile.GFile(dataset_name, mode="r").read()
    print("[*] Reading all contexts")

    if not os.path.exists(vocab_fname):
        print("[*] Create vocab from %s to %s ..." % (dataset_name, vocab_fname))
        create_vobabulary(vocab_fname, context, vocab_size)
    else:
        print("[*] Skip creating vocab")
    print("[*] Convert data into vocab indicies..." )
    questions_to_token_ids(vocab_fname, vocab_size)

if __name__ == '__main__':
    dataset_name = 'collect.data'
    prepare_data(dataset_name, 30000)
