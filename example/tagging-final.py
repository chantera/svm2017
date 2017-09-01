#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import functions as F, links as L
import numpy as np
from teras import dataset, io, logging, preprocessing, training
from teras.app import App, arg
import teras.framework.chainer as framework_utils


class CharCNN(chainer.Chain):

    def __init__(self, embedding, out_size=None,
                 window_size=3, dropout=0.5):
        super().__init__()
        if isinstance(embedding, np.ndarray):
            vocab_size, embed_size = embedding.shape
        else:
            vocab_size, embed_size = embedding
            embedding = None
        if out_size is None:
            out_size = embed_size
        self.out_size = out_size
        with self.init_scope():
            self.embed = L.EmbedID(
                in_size=vocab_size,
                out_size=embed_size,
                initialW=embedding,
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
        self._desc = {
            'embedding': (vocab_size, embed_size),
            'out_size': out_size,
            'window_size': window_size,
            'dropout': dropout,
        }

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

    def __repr__(self):
        return repr(self._desc)


class Model(chainer.Chain):

    def __init__(self,
                 word_embedding,
                 char_embedding,
                 n_labels,
                 char_window_size=3,
                 char_filter_size=30,
                 n_blstm_layers=1,
                 lstm_hidden_size=200,
                 dropout=0.5):
        super().__init__()
        if isinstance(word_embedding, np.ndarray):
            word_vocab_size, word_embed_size = word_embedding.shape
        else:
            word_vocab_size, word_embed_size = word_embedding
            word_embedding = None
        self.dropout = dropout
        with self.init_scope():
            self.word_embed = L.EmbedID(word_vocab_size, word_embed_size,
                                        word_embedding)
            self.char_cnn = CharCNN(char_embedding,
                                    char_filter_size, char_window_size,
                                    dropout)
            self.blstm = L.NStepBiLSTM(n_blstm_layers,
                                       word_embed_size + char_filter_size,
                                       lstm_hidden_size, dropout)
            self.linear = L.Linear(lstm_hidden_size * 2, n_labels)
        self._desc = {
            'word_embedding': (word_vocab_size, word_embed_size),
            'n_labels': n_labels,
            'n_blstm_layers': n_blstm_layers,
            'lstm_hidden_size': lstm_hidden_size,
            'links': {
                'word_embedding': self.word_embed,
                'char_cnn': self.char_cnn,
                'blstm': self.blstm,
                'linear': self.linear,
            },
            'dropout': dropout,
        }

    def __call__(self, word_seqs, char_seqs):
        indices = [0]
        xs = []
        for word_seq, char_seq in zip(word_seqs, char_seqs):
            x_word = self.word_embed(self.xp.array(word_seq))
            x_char = self.char_cnn(char_seq)
            x = F.concat((x_word, x_char))
            indices.append(indices[-1] + x.shape[0])
            xs.append(F.dropout(x, self.dropout))
        hy, cy, ys = self.blstm(hx=None, cx=None, xs=xs)
        ys = F.concat(ys, axis=0)
        ys = self.linear(F.dropout(ys, self.dropout))
        ys = F.split_axis(ys, indices[1:-1], axis=0)
        return ys

    def __repr__(self):
        return repr(self._desc)


class DataLoader(dataset.loader.CorpusLoader):

    def __init__(self,
                 word_embed_size=100,
                 char_embed_size=30,
                 word_embed_file=None,
                 word_preprocess=lambda x: x.lower(),
                 word_unknown="<UNKNOWN>",
                 embed_dtype='float32'):
        super(DataLoader, self).__init__(reader=io.reader.ConllReader())
        self.use_pretrained = word_embed_file is not None
        self.add_processor(
            'word', embed_file=word_embed_file, embed_size=word_embed_size,
            embed_dtype=embed_dtype,
            preprocess=word_preprocess, unknown=word_unknown)
        self.add_processor(
            'char', embed_file=None, embed_size=char_embed_size,
            embed_dtype=embed_dtype,
            preprocess=lambda x: x.lower())
        self.tag_map = preprocessing.text.Vocab()

    def map(self, item):
        # item -> (words, chars, labels)
        words, labels = zip(*[(token['form'],
                               self.tag_map.add(token['postag']))
                              for token in item[1:]])  # skip root token
        sample = (self._word_transform(words, as_one=True),
                  self.get_processor('char')
                  .fit_transform(words, as_one=False),
                  np.array(labels, dtype=np.int32))
        return sample

    def load(self, file, train=False, size=None):
        if train and not self.use_pretrained:
            # assign an index if the given word is not in vocabulary
            self._word_transform = self.get_processor('word').fit_transform
        else:
            # return the unknown word index if the word is not in vocabulary
            self._word_transform = self.get_processor('word').transform
        return super(DataLoader, self).load(file, train, size)


def train(
        train_file,
        test_file,
        word_embed_file=None,
        word_embed_size=100,
        char_embed_size=30,
        n_epoch=20,
        batch_size=10,
        lr=0.01,
        gpu=-1,
        seed=None):
    if seed is not None:
        import random
        import numpy
        random.seed(seed)
        numpy.random.seed(seed)
        if gpu >= 0:
            try:
                import cupy
                cupy.cuda.runtime.setDevice(gpu)
                cupy.random.seed(seed)
            except Exception as e:
                logging.error(str(e))
        logging.info("random seed: {}".format(seed))
    framework_utils.set_debug(App.debug)

    # Load files
    logging.info('initialize DataLoader with word_embed_file={}, '
                 'word_embed_size={}, and char_embed_size={}'
                 .format(word_embed_file, word_embed_size, char_embed_size))
    loader = DataLoader(word_embed_size, char_embed_size, word_embed_file)
    logging.info('load train dataset from {}'.format(train_file))
    train_dataset = loader.load(train_file, train=True)
    if test_file:
        logging.info('load test dataset from {}'.format(test_file))
        test_dataset = loader.load(test_file, train=False)
    else:
        test_dataset = None

    model = Model(word_embedding=loader.get_embeddings('word'),
                  char_embedding=loader.get_embeddings('char'),
                  n_labels=len(loader.tag_map))
    logging.info('initialized model: {}'.format(model))
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam(lr)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))

    def compute_loss(ys, ts):
        ys, ts = F.concat(ys, axis=0), F.concat(ts, axis=0)
        if gpu >= 0:
            ts.to_gpu()
        return F.softmax_cross_entropy(ys, ts, ignore_label=-1)

    def compute_accuracy(ys, ts):
        ys, ts = F.concat(ys, axis=0), F.concat(ts, axis=0)
        if gpu >= 0:
            ts.to_gpu()
        return F.accuracy(ys, ts, ignore_label=-1)

    trainer = training.Trainer(optimizer, model,
                               loss_func=compute_loss,
                               accuracy_func=compute_accuracy)
    trainer.configure(framework_utils.config)

    trainer.fit(train_dataset, None,
                batch_size=batch_size,
                epochs=n_epoch,
                validation_data=test_dataset,
                verbose=App.verbose)


if __name__ == "__main__":
    logging.AppLogger.configure(mkdir=True)
    App.add_command('train', train, {
        'train_file': arg('--trainfile', type=str, required=True),
        'test_file': arg('--testfile', type=str),
        'word_embed_file': arg('--embedfile', type=str),
        'n_epoch': arg('--epoch', type=int, default=20),
        'batch_size': arg('--batchsize', type=int, default=10),
        'lr': arg('--lr', type=float, default=0.01),
        'gpu': arg('--gpu', type=int, default=-1),
        'seed': arg('--seed', type=int, default=1),
    })
    chainer.config.debug = False
    chainer.config.type_check = False
    App.run()
