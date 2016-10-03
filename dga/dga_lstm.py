from __future__ import division, print_function

import sys
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from random import shuffle, seed

import stf_dataset
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Masking, Dropout
from keras.layers.recurrent import GRU
from keras.layers import Convolution1D, MaxPooling1D, Flatten
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from unbalanced_dataset import OverSampler, UnderSampler
import numpy as np


def filter_by_string(df, col, string):
    '''
    Filters a dataframe column by a string.
    '''
    return df[df[col].str.contains(string, regex=False) == True]


def load_csv_data(dataset_file):
    with open(dataset_file, 'rb') as csvfile:
        rawreader = pd.read_csv(csvfile, delimiter='|', names=[
                                "note", "label", "model_id", "state"],
                                skipinitialspace=True)
        # pd.core.strings.str_strip(rawreader['note'])
        # pd.core.strings.str_strip(rawreader['label'])
        # pd.core.strings.str_strip(rawreader['model_id'])
        # pd.core.strings.str_strip(rawreader['state'])

    if len(rawreader) is 0:
        return

    return rawreader


def split_data(data, split_pct=0.1):
        '''
        Splits data into training and testing.
        '''
        return train_test_split(data, test_size=split_pct)


def build_lstm(input_shape):
    model = Sequential()
    # model.add(Masking(input_shape=input_shape, mask_value=-1.))
    model.add(Embedding(input_shape[0], 128, input_length=input_shape[1]))

    model.add(Convolution1D(nb_filter=64,
                            filter_length=5,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=model.output_shape[1]))

    model.add(Flatten())

    model.add(Dense(128))

    # model.add(GRU(128, return_sequences=False))
    # Add dropout if overfitting
    # model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model(model, x_train, y_train, batch_size, checkpointer, epochs=20):
    print("Training model...")

    # FIT THE MODEL
    model.fit(x_train, y_train, nb_epoch=epochs, batch_size=batch_size,
              validation_split=0.2,
              verbose=1, callbacks=[checkpointer], shuffle=True)


def test_model(model, x_test, y_test):
    test_preds = model.predict_classes(x_test, len(x_test), verbose=1)
    print ("Testing Dataset) ", metrics.confusion_matrix(y_test, test_preds))


if __name__ == "__main__":
    if(len(sys.argv) < 2):
        print('need to specify csv raw dataset filename as argument.')
        sys.exit(1)
    filename = sys.argv[1]
    df = load_csv_data(filename)

    if df is None:
        sys.exit(1)

    # FILTER by Protocol and then Category
    normal_data = filter_by_string(filter_by_string(df, 'label', 'UDP'),
        'label', 'Normal')['state'].values.tolist()
    botnet_data = filter_by_string(filter_by_string(df, 'label', 'UDP'),
        'label', 'Botnet')['state'].values.tolist()

    # Set 0 or 1 depending on the sample Category
    y_data = [0 for i in xrange(len(normal_data))] + [1 for i in xrange(len(botnet_data))]

    # Make sure this is right
    assert len(normal_data) > 0 and len(botnet_data) > 0
    assert len(normal_data) + len(botnet_data) == len(y_data)

    data = normal_data + botnet_data
    data = [x[3:] for x in data if len(x) > 3]
    print ("normal", len(normal_data))
    print ("botnet", len(botnet_data))

    # Split sequences with spaces every 5 characters to convert them to words
    n = 5
    text = []
    for x in data:
        text.append(" ".join([x[i:i+n] for i in range(0, len(x), n)]))

    assert len(text) == len(data)
    tokenizer = Tokenizer(filters=stf_dataset.text_filter(), lower=False)
    tokenizer.fit_on_texts(text)
    seq = tokenizer.texts_to_sequences(text)
    print("text - seq", len(text), len(seq))
    print (tokenizer.word_index)
    print (len(max(seq, key=len)))
    mat = sequence.pad_sequences(seq, maxlen=500)
    _, max_word_index = max(tokenizer.word_index.iteritems(), key=lambda x:x[1])
    print("max word index", max_word_index)
    raw_input("..")
    assert len(data) == len(seq)
    data = zip(mat, y_data)

    # shuffle
    seed(1)
    shuffle(data)

    # split into training and testing
    i = int(len(data)*0.8)
    print(i)
    train_data = data[:i]
    test_data = data[i:]
    print (len(data), len(train_data), len(test_data))
    raw_input("..")
    # train_data, test_data = split_data(data, split_pct=0.2)
    train_x_data, train_y_data = zip(*train_data)

    # set the max number of steps (max length of sequences)
    maxlen = min(len(max(train_x_data, key=len)), 500)

    # vectorize the data to feed the net
    # train_x_data, train_y_data = stf_dataset.vectorize(train_x_data,
    #    train_y_data, mode='int', sampling='OverSampler', maxlen=maxlen, minlen=maxlen, start_offset=5)

    # build the net
    model = build_lstm(input_shape=( max_word_index+1, maxlen))
    filename = '/tmp/weights_latest.hdf5'
    # a checkpointer to save the best trained model
    checkpointer = ModelCheckpoint(filepath=filename,
                                verbose=1, save_best_only=True)

    # reshape for Embedding
    # train_x_data = np.reshape(train_x_data, (len(train_x_data), train_x_data.shape[1] * train_x_data.shape[2]))

    train_x_data = [np.asarray(x, dtype=np.float32) for x in train_x_data]
    train_x_data = np.asarray(train_x_data, dtype=np.float32)
    train_y_data = np.asarray(train_y_data)

    train_x_data, train_y_data = OverSampler(verbose=True).fit_transform(train_x_data, train_y_data)

    print (train_x_data.shape)
    print("normal", len(train_y_data[train_y_data==0]))
    print("botnet", len(train_y_data[train_y_data==1]))

    # train the model
    train_model(model, train_x_data, train_y_data,
            checkpointer=checkpointer, batch_size=32, epochs=15)

    # test_x_data, test_y_data = stf_dataset.vectorize(test_x_data, test_y_data, mode='int',
    #                                                 maxlen=maxlen, minlen=0)

    # reshape for Embedding
    # test_x_data = np.reshape(test_x_data, (len(test_x_data), test_x_data.shape[1] * test_x_data.shape[2]))

    # test the model
    # test_x_data, test_y_data = zip(*test_data)

    # Convert test data to numpy arrays
    test_x_data, test_y_data = zip(*test_data)
    test_x_data = [np.asarray(x, dtype=np.float32) for x in test_x_data]
    test_x_data = np.asarray(test_x_data, dtype=np.float32)
    test_y_data = np.asarray(test_y_data)
    print (test_x_data.shape, test_y_data.shape)
    print("normal", len(test_y_data[test_y_data==0]))
    print("botnet", len(test_y_data[test_y_data==1]))

    # load trained weights
    model.load_weights(filename)

    test_model(model, test_x_data, test_y_data)
