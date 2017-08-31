#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from collections import defaultdict
import logging

import chainer
from chainer import functions as F, links as L
import numpy as np


class CharCNN(chainer.Chain):

    def __init__(self, vocab_size, embed_size,
                 out_size=None, window_size=3, dropout=0.5):
        super().__init__()
        if out_size is None:
            out_size = embed_size
        self.out_size = out_size
        with self.init_scope():
            self.embed = L.EmbedID(
                in_size=vocab_size,
                out_size=embed_size,
            )
            self.conv = L.Convolution2D(
                in_channels=1,
                out_channels=out_size,
                ksize=(window_size, embed_size),
                stride=(1, embed_size),
                pad=(int(window_size / 2), 0),
                nobias=True,
                initialW=None,
            )
        self.dropout = dropout

    def __call__(self, chars):
        """
        Note: we use zero-padding instead of spacial PADDING token.
        """
        if isinstance(chars, (tuple, list)):
            return F.vstack([self.forward_one(_chars) for _chars in chars])
        return self.forward_one(chars)

    def forward_one(self, chars):
        x = self.embed(self.xp.array(chars))
        x = F.dropout(x, self.dropout)
        h, w = x.shape
        C = self.conv(F.reshape(x, (1, 1, h, w)))
        C = F.transpose(C, (0, 3, 2, 1))  # (1, 1, len, dim)
        M = F.max_pooling_2d(C, ksize=(h, 1), stride=None, pad=0)
        y = F.reshape(M, (self.out_size,))
        return y


class Model(chainer.Chain):

    def __init__(self,
                 word_vocab_size,
                 word_embed_size,
                 char_vocab_size,
                 char_embed_size,
                 n_labels,
                 char_window_size=3,
                 char_filter_size=30,
                 n_blstm_layers=1,
                 lstm_hidden_size=200,
                 dropout=0.5):
        super().__init__()
        self.dropout = dropout
        with self.init_scope():
            self.word_embed = L.EmbedID(word_vocab_size, word_embed_size)
            self.char_cnn = CharCNN(char_vocab_size, char_embed_size,
                                    char_filter_size, char_window_size,
                                    dropout)
            self.blstm = L.NStepBiLSTM(n_blstm_layers,
                                       word_embed_size + char_filter_size,
                                       lstm_hidden_size, dropout)
            self.linear = L.Linear(lstm_hidden_size * 2, n_labels)

    def __call__(self, word_seqs, char_seqs):
        indices = [0]
        xs = []
        for word_seq, char_seq in zip(word_seqs, char_seqs):
            x_word = self.word_embed(word_seq)
            x_char = self.char_cnn(char_seq)
            x = F.concat((x_word, x_char))
            indices.append(indices[-1] + x.shape[0])
            xs.append(F.dropout(x, self.dropout))
        hy, cy, ys = self.blstm(hx=None, cx=None, xs=xs)
        ys = F.concat(ys, axis=0)
        ys = self.linear(F.dropout(ys, self.dropout))
        ys = F.split_axis(ys, indices[1:-1], axis=0)
        return ys


def read_conll(file):
    sentences = []
    tokens = []
    for line in open(file, mode='r'):
        line = line.strip()
        if not line:
            if len(tokens) > 0:
                sentences.append(tokens)
                tokens = []
        else:
            cols = line.split("\t")
            token = {'form': cols[1], 'postag': cols[4]}
            tokens.append(token)
    if len(tokens) > 0:
        sentences.append(tokens)
    return sentences


def main(train_file,
         test_file,
         n_epoch=20,
         batch_size=10,
         lr=0.01,
         gpu=-1):
    train_sentences = read_conll(train_file)
    n_train_samples = len(train_sentences)
    test_sentences = read_conll(test_file)
    n_test_samples = len(test_sentences)

    word2id = defaultdict(lambda: len(word2id))
    char2id = defaultdict(lambda: len(char2id))
    tag2id = defaultdict(lambda: len(tag2id))

    unknown = "<UNKNOWN>"
    unknown_word_id = word2id[unknown]
    unknown_char_id = char2id[unknown]

    train_words = []
    train_chars = []
    train_tags = []
    for tokens in train_sentences:
        words = []
        chars = []
        tags = []
        for token in tokens:
            words.append(word2id[token['form'].lower()])
            chars.append(np.array([char2id[char]
                                   for char in token['form']], np.int32))
            tags.append(tag2id[token['postag'].lower()])
        train_words.append(np.array(words, np.int32))
        train_chars.append(chars)
        train_tags.append(np.array(tags, np.int32))

    test_words = []
    test_chars = []
    test_tags = []
    for tokens in test_sentences:
        words = []
        chars = []
        tags = []
        for token in tokens:
            words.append(word2id.get(token['form'].lower(),
                                     unknown_word_id))
            chars.append(np.array([char2id.get(char, unknown_char_id)
                                   for char in token['form']], np.int32))
            tags.append(tag2id[token['postag'].lower()])
        test_words.append(np.array(words, np.int32))
        test_chars.append(chars)
        test_tags.append(np.array(tags, np.int32))

    model = Model(word_vocab_size=len(word2id),
                  word_embed_size=100,
                  char_vocab_size=len(char2id),
                  char_embed_size=30,
                  n_labels=len(tag2id))
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam(lr)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))

    indices = np.arange(n_train_samples)
    for epoch in range(n_epoch):
        iteration, epoch_loss, epoch_accuracy = 0, 0.0, 0.0
        chainer.config.train = True
        chainer.config.enable_backprop = True
        np.random.shuffle(indices)
        for i in range(0, n_train_samples, batch_size):
            model.cleargrads()
            words = np.take(train_words, indices[i: i + batch_size])
            chars = np.take(train_chars, indices[i: i + batch_size])
            tags = np.take(train_tags, indices[i: i + batch_size])
            ys = model(words, chars)
            ys = F.concat(ys, axis=0)
            ts = F.concat(tags, axis=0)
            loss = F.softmax_cross_entropy(ys, ts, ignore_label=-1)
            loss.backward()
            optimizer.update()
            accuracy = F.accuracy(ys, ts, ignore_label=-1)
            epoch_loss += loss.data
            epoch_accuracy += accuracy.data
            iteration += 1
        logging.info('[train epoch {:02d}] loss: {:.6f}, accuracy: {:.6f}'
                     .format(epoch + 1,
                             epoch_loss / iteration,
                             epoch_accuracy / iteration))

        model.cleargrads()
        iteration, epoch_loss, epoch_accuracy = 0, 0.0, 0.0
        chainer.config.train = False
        chainer.config.enable_backprop = False
        for i in range(0, n_test_samples, batch_size):
            words = test_words[i: i + batch_size]
            chars = test_chars[i: i + batch_size]
            tags = test_tags[i: i + batch_size]
            ys = model(words, chars)
            ys = F.concat(ys, axis=0)
            ts = F.concat(tags, axis=0)
            loss = F.softmax_cross_entropy(ys, ts, ignore_label=-1)
            accuracy = F.accuracy(ys, ts, ignore_label=-1)
            epoch_loss += loss.data
            epoch_accuracy += accuracy.data
            iteration += 1
        logging.info('[test  epoch {:02d}] loss: {:.6f}, accuracy: {:.6f}'
                     .format(epoch + 1,
                             epoch_loss / iteration,
                             epoch_accuracy / iteration))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    chainer.config.debug = False
    chainer.config.type_check = False
    parser = ArgumentParser()
    parser.add_argument('--trainfile', type=str, required=True)
    parser.add_argument('--testfile', type=str)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batchsize', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gpu', type=int, default=-1)
    args = parser.parse_args()
    main(args.trainfile, args.testfile, args.epoch, args.batchsize, args.lr)
