#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 14:50:42 2018

@author: elie
"""

# Millions of questions are asked every year about drugs. There is a limited number of query types but the same question could be asked in many ways. Therefore, understanding what information people expect when asking a question is a great challenge. The goal of Posos’ challenge is to predict for each question its associated intent.
#
# The input data (N=8028) is a French written list of questions. Each line consists of a unique ID followed by one question. the question came from different source like : web scraping like doctisimo, doctor, pharmacist. They correspond to:
#
#    ID: line number. Relates to the line number in the output file.
#    Question : the question whose intent has to be predicted.
#
# Below is an example of the input file content:
#
# 1;Est-ce qu'il existe une forme adaptée aux enfants de 5ans du Micropakine ?
# 2;laroxyl à doses faibles pour le stress ?
# The output file contains the intent associated to each ID.
# They correspond to the line number and the intent identification number.
# ID;intention
# 1;44
# 2;31
#
#
# Intents are homogeneously distributed between training and test files (N=2035).
#
# Questions must be classified according to a list of 52 different intent categories. Each intent matches an anonymized id number between 0 and 51 distributed as follows:
#
#    a list of 50 predefined intents
#    a "multiple" category (n°39), used for questions with several intents
#    a "other" category (n°1), for questions that don't match any of the 50 predefined intents
#
# Métrique : Accuracy


from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from keras.layers import CuDNNGRU
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Embedding, Flatten, Dense,  Conv2D, MaxPool2D
from keras.layers import Dropout, Input, Concatenate
from keras.layers import Reshape
import numpy as np
import pandas as pd
import re, string
from itertools import chain
from sklearn.utils import class_weight

from keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow import keras
import os
from keras.utils import to_categorical
layers = keras.layers



def embd(embedding_dim, dir):
    embeddings_index = {}
    f = open(os.path.join(dir), encoding="utf-8")
    word = []
    for line in f:
        values = line.split()
        word = values[0]
        if is_number_tryexcept(values[1]):
            coefs = np.asarray(values[1:], dtype='float64')
            embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    embedding_matrix_wv = np.zeros((len(word_index), embedding_dim))
    word_not_in = []
    word_in = []
    print(embedding_matrix_wv.shape)
    for word, i in word_index.items():
        if i < len(word_index):
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is None:
                word_not_in = word_not_in + [word]
                # Whether or not the input value 0 is a special "padding" value that should be masked out.
                # This is useful when using recurrent layers which may take variable length input.
                # If this is True then all subsequent layers in the model need to support masking or an exception
                # will be raised. If mask_zero is set to True, as a consequence, index 0 cannot be used in the vocabulary
                # (input_dim should equal size of vocabulary + 1).
                # embedding_matrix[i]= np.random.normal(size= embedding_dim)
            else:
                embedding_matrix_wv[i] = embedding_vector
                word_in = word_in + [word]

    print(len(word_not_in))
    print(len(word_in))
    return embedding_matrix_wv


def score_function(y_true, y_pred):
    """ Returns the score """
    score = 0
    length1 = y_true.shape[0]
    for i in range(length1):
        if y_pred[i] == y_true[i]:
            score += 1
    return float(score) / float(length1)


def is_number_tryexcept(s):
    """ Returns True is string is a number. """
    try:
        float(s)
        return True
    except ValueError:
        return False


def count_in_vocab(text, vocab):
    z = 0
    for i in text.split(' '):
        if i in vocab:
            z = z + 1
    return z


def clean_text(text, stop=False, transforme_number=True):
    """ Returns clean text """
    translator = str.maketrans("!\"#$%&\'()*,.//:;<=>?@[\\]^_`{|}~", "                               ")
    ## Remove puncuation
    text = text.translate(translator)
    text = re.sub(r"-", " ", text)
    text = re.sub(r"®", " ", text)
    text = re.sub(r"©", " ", text)
    text = re.sub(r"™", " ", text)
    text = re.sub(r"’", " ’ ", text)
    text = re.sub(r"´", " ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"°", " ", text)
    text = re.sub(r"»", " ", text)
    text = re.sub(r"\\xa0", " ", text)
    text = re.sub(r"«", " ", text)
    text = re.sub(r"•", " ", text)
    text = re.sub(r"€", "euros", text)
    text = re.sub(r"…", " ", text)
    text = re.sub(r"2ndaire", "secondaire", text)

    ## Convert words to lower case
    text = text.lower()
    stops = ['']
    ## Remove stop words ?
    if stop == True:
        stops = list(string.ascii_lowercase) + stopwords.words("french") + ['+', 'les', 'ca', 'sous', 'ci', 'petite']

    ##Add space between numbers
    l1 = [re.split('(\d+)', w.strip()) for w in text.split(' ') if not w in stops]
    text = list(chain(*l1))
    text = [w.strip() for w in text if not w in stops]
    text = " ".join(text)
    if transforme_number == True:
        text = re.sub(r"[0-9]+", "nombre", text)
    else:
        text = re.sub(r"[0-9]+", "", text)
    return text


def MODEL_WV(embedding_dim=500):
    inp = Input(shape=(max_seq_len,))
    x = Embedding(len(word_index), embedding_dim, weights=[embedding_matrix_wv], trainable=True)(inp)
    x = Dropout(0.9)(x)
    x = CuDNNGRU(300, return_sequences=False)(x)
    x = Dropout(0.5)(x)
    x = Dense(51, activation="softmax")(x)
    model = Model(inputs=inp, outputs=x)
    adam = Adam(clipnorm=1.)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


def MODEL_frWac(embedding_dim=500):
    inp = Input(shape=(max_seq_len,))
    x = Embedding(len(word_index), embedding_dim, weights=[embedding_matrix_frWac], trainable=True)(inp)
    x = Dropout(0.9)(x)
    x = CuDNNGRU(300, return_sequences=False)(x)
    x = Dropout(0.5)(x)
    x = Dense(51, activation="softmax")(x)
    model = Model(inputs=inp, outputs=x)
    adam = Adam(clipnorm=1.)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


def MODEL_wiki(embedding_dim=300):
    inp = Input(shape=(max_seq_len,))
    x = Embedding(len(word_index), embedding_dim, weights=[embedding_matrix_wiki], trainable=True)(inp)
    x = Dropout(0.9)(x)
    x = CuDNNGRU(300, return_sequences=False)(x)
    x = Dropout(0.5)(x)
    x = Dense(51, activation="softmax")(x)
    model = Model(inputs=inp, outputs=x)
    adam = Adam(clipnorm=1.)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


filter_sizes = [1, 2, 1]
num_filters = 300
drop = 0.6


def model_cv_wv(embedding_dim=300):
    # this returns a tensor
    print("Creating Model...")
    inputs = Input(shape=(max_seq_len,), dtype='int32')
    embedding = Embedding(input_dim=len(word_index), output_dim=embedding_dim, input_length=max_seq_len,
                          weights=[embedding_matrix_wiki], trainable=True)(inputs)
    reshape = Reshape((max_seq_len, embedding_dim, 1))(embedding)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(max_seq_len - filter_sizes[0] + 0, 1), strides=(1, 1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(max_seq_len - filter_sizes[1] + 0, 1), strides=(1, 1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(max_seq_len - filter_sizes[2] + 0, 1), strides=(1, 1), padding='valid')(conv_2)
    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])

    # concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=51, activation='softmax')(dropout)

    # this creates a model that includes
    model = Model(inputs=inputs, outputs=output)

    # checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=1.)
    # adam = adam(clipnorm=1.)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def model_cv_wiki(embedding_dim=500):
    # this returns a tensor
    print("Creating Model...")
    inputs = Input(shape=(max_seq_len,), dtype='int32')
    embedding = Embedding(input_dim=len(word_index), output_dim=embedding_dim, input_length=max_seq_len,
                          weights=[embedding_matrix_wv], trainable=True)(inputs)
    reshape = Reshape((max_seq_len, embedding_dim, 1))(embedding)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(max_seq_len - filter_sizes[0] + 0, 1), strides=(1, 1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(max_seq_len - filter_sizes[1] + 0, 1), strides=(1, 1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(max_seq_len - filter_sizes[2] + 0, 1), strides=(1, 1), padding='valid')(conv_2)
    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])

    # concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=51, activation='softmax')(dropout)

    # this creates a model that includes
    model = Model(inputs=inputs, outputs=output)

    # checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=1.)
    # adam = adam(clipnorm=1.)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def model_emd_normal(embedding_dim=600):
    filter_sizes = [1, 2, 1]
    # this returns a tensor
    print("Creating Model...")
    inputs = Input(shape=(max_seq_len,), dtype='int32')
    embedding = Embedding(input_dim=len(word_index), output_dim=embedding_dim, input_length=max_seq_len,
                          embeddings_initializer='RandomNormal',
                          # weights=[embedding_matrix_wv],
                          trainable=True)(inputs)
    reshape = Reshape((max_seq_len, embedding_dim, 1))(embedding)

    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(max_seq_len - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(max_seq_len - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(max_seq_len - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(conv_2)
    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])

    # concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=51, activation='softmax')(dropout)

    # this creates a model that includes
    model = Model(inputs=inputs, outputs=output)

    # checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=1.)
    # adam = adam(clipnorm=1.)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#Read the train data and the labels

X_train = pd.read_csv('Data/input_train.csv', sep=';', index_col=False)
y_train = pd.read_csv('Data/challenge_output_data_training_file_predict_the_expected_answer.csv', sep=';', index_col=False)
# cleaning the text in train with the function clean_text
X_train['question'] = X_train['question'].map(lambda x: clean_text(x, stop=False, transforme_number=True))

y_train=y_train.intention

#Read a dictionnairie to replace uncorrect word by correct word.
# The correction is done with google scheet.

dic_correction = pd.read_csv('Data/correction_mots_avec_accent.csv', sep=';', index_col=False)
dic_correction.A_remplacer = dic_correction.A_remplacer.map(lambda x: clean_text(str(x)))
correction_of_word = dict(zip(dic_correction.A_corriger, dic_correction.A_remplacer))
d2 = {r'(\b){}(\b)'.format(k): r'\1{}\2'.format(v) for k, v in correction_of_word.items()}
X_train = X_train.replace({'question': d2}, regex=True)

# Same as train : Read,  clean and correct the data

X_test = pd.read_csv('Data/input_test.csv', sep=';', index_col=False)
X_test['question'] = X_test['question'].map(lambda x: clean_text(x, stop=False))
X_test = X_test.replace({'question': d2}, regex=True)
ID = X_test.ID


#class_weights = class_weight.compute_class_weight('balanced', np.unique(lab), lab)
#class_weights_dic = dict(zip(np.unique(lab), class_weights))
#class_weights_exp = np.exp(class_weights)

all_text = X_train.append(X_test)
max_seq_len = 120
vocab_size = 8948

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(all_text.question)

x_train = tokenizer.texts_to_sequences(X_train.question)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_seq_len)

x_test = tokenizer.texts_to_sequences(X_test.question)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_seq_len)

word_index = tokenizer.word_index



embedding_dim = 500
embedding_dim2 = 300
w2v_dir = 'embeddings/corpus_train_test_correction_word_dic_min_coun_1_iter_5000_w2v_sg1_size_500.txt'

embedding_matrix_wv = embd(embedding_dim, w2v_dir)

#http://fauconnier.github.io/
w2v_2_dir = 'embeddings/frWac.txt'

embedding_matrix_frWac = embd(embedding_dim, w2v_2_dir)

#https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.fr.vec
wiki_dir = 'embeddings/wiki.fr/wiki.fr.vec'

embedding_matrix_wiki = embd(embedding_dim2, wiki_dir)

num_epochs = 50
batch_size = 102
##
ppl1 = Pipeline([
    ("MODEL_WV", KerasClassifier(MODEL_WV, epochs=num_epochs + 50,
                                 batch_size=batch_size, verbose=1, shuffle=False))])

ppl2 = Pipeline([
    ("MODEL_wiki", KerasClassifier(MODEL_wiki, epochs=num_epochs + 50,
                                   batch_size=batch_size, verbose=1, shuffle=False))])

ppl3 = Pipeline([
    ("model_cv_wv", KerasClassifier(model_cv_wv, epochs=num_epochs + 30,
                                    batch_size=batch_size, verbose=1, shuffle=False))])

ppl4 = Pipeline([
    ("model_cv_wiki", KerasClassifier(model_cv_wiki, epochs=num_epochs + 30,
                                      batch_size=batch_size, verbose=1, shuffle=False))])
ppl5 = Pipeline([
    ("model_emd_normal", KerasClassifier(model_emd_normal, epochs=num_epochs + 40,
                                         batch_size=batch_size, verbose=1, shuffle=False))])

ppl6 = Pipeline([
    ("MODEL_frWac", KerasClassifier(MODEL_frWac, epochs=num_epochs + 50,
                                    batch_size=batch_size, verbose=1, shuffle=False))])

eclf = VotingClassifier(
    estimators=[('MODEL_WV', ppl1), ('MODEL_wiki', ppl2), ('model_cv_wv', ppl3), ('model_cv_wiki', ppl4),
                ('model_emd_normal', ppl5), ('MODEL_frWac', ppl6)], voting='soft',
    weights=[1.0, 4.0, 2.0, 3.0, 2.0, 3.0])

eclf = eclf.fit(x_train, y_train)
eclf.score(x_train, y_train)

y_test_pred = eclf.predict(x_test)
sub = pd.DataFrame(np.column_stack((ID, y_test_pred)), columns=['ID', 'intention'])
sub.to_csv('submit/5Models_12_04_2019_.csv', index=False, sep=';')
