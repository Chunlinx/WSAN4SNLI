# coding: utf-8
import time
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pickle
import math
import os
import sys
import logging
import utils

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("models")


def weight_variable(shape, name_prefix):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable(name_prefix + "_weights", initializer=initial)

def xavier_weights_variable(shape, name_prefix):
    if len(shape) == 4:
        scale = np.sqrt(6.0 / (shape[0]*shape[1]*shape[2] + shape[-1]))
    else:
        scale = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape, -scale, scale, dtype=tf.float32)
    weights = tf.get_variable(name_prefix + "_weighs", initializer=initial)

    return weights


def bias_variable(shape, name_prefix):
    initial = tf.constant(0.0001, shape=shape)
    return tf.get_variable(name_prefix + "_bias", initializer=initial)


def identical(x):
    return x


def s_tanh(x):
    return 1.7159 * tf.nn.tanh(2.0 * x / 3.0)


def word_level_lstm(inputs, sentence_length, dims_hidden_unit, keep_prob, batch_size):
    '''
    输入为x序列和对应的mask序列,根据其他的变量来进行lstm的计算
    输入的两个序列的shape: (nb_samples, sentence_length, dim_inputs) & (nb_samples, sentence_leng th)
    :param inputs: 输入x_seq
    :param sentence_length: 句子长度
    :param dims_hidden_unit: 隐层维度，是一个列表，数量表示lstm的层数
    :param keep_prob: 用于lstm层的dropout
    :param batch_size: 批次大小
    :type batch_size: tf.tensor
    # :param mode: 模式，表示最后得到句子表示的方式: "ave", "last", "sa"
    :return: 返回的是输出序列,具体的句子表示要如何得到，不在这个函数中进行
    '''
    cells = []
    for dim_hidden in dims_hidden_unit:
        lstm_cell = rnn.BasicLSTMCell(num_units=dim_hidden, forget_bias=1.0)
        lstm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)   # 一般dropout只加在输出的时候
        cells.append(lstm_cell)

    multi_lstm_cell = rnn.MultiRNNCell(cells)
    initial_state = multi_lstm_cell.zero_state(batch_size, dtype=tf.float32)
    state = initial_state
    outputs = []
    with tf.variable_scope("lstm"):
        for step in range(sentence_length):
            if step > 0:
                tf.get_variable_scope().reuse_variables()
            (cell_output, state) = multi_lstm_cell(inputs[:, step, :], state)
            outputs.append(cell_output)

    # weights = tf.nn.softmax((mask + 10.0) * mask)   # shape: nb_samples, sentence_length
    outputs = tf.stack(outputs, axis=1)  # shape: nb_samples, sentence_length, dim
    # sentence_pre = tf.reduce_sum(outputs * tf.expand_dims(weights, axis=-1), axis=1)  # shape: nb_samples, dim

    return outputs
    # 返回的是序列输出


def word_level_blstm(inputs, sentence_length, dim_hidden_unit, keep_prob, batch_size):
    '''
    双向的lstm,参数情况同lstm一样。只支持单层的，多层的需要多次调用这个函数
    :param inputs: shape: (nb_samples, sentence_length, dim_inputs)
    :param mask:
    :param sentence_length:
    :param dim_hidden_unit: 只允许是一个值，不允许是一个列表
    :param keep_prob:
    :param batch_size:
    :return:
    '''
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # 为了满足static_bidirectional_rnn用输入形状的不同要求
    x = tf.unstack(inputs, sentence_length, 1)
    fw_lstm = rnn.BasicLSTMCell(num_units=dim_hidden_unit, forget_bias=1.0)
    fw_lstm = rnn.DropoutWrapper(fw_lstm, output_keep_prob=keep_prob)
    bw_lstm = rnn.BasicLSTMCell(num_units=dim_hidden_unit, forget_bias=1.0)
    bw_lstm = rnn.DropoutWrapper(bw_lstm, output_keep_prob=keep_prob)
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(fw_lstm, bw_lstm, x, dtype=tf.float32)
    except Exception:  # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(fw_lstm, bw_lstm, x, dtype=tf.float32)
    # outputs: list of (batch_size, 2*dim_hidden_unit)
    outputs = tf.stack(outputs, axis=1)     # (batch_size, sentence_length, 2*dim_hidden_unit)

    return outputs


def my_lstm_layer(inputs, hidden_size, keep_prob, sequence_length, out="AVERAGE", mask=None):
    # mask: (None, sequence_length)
    lstm_cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    dropout_letm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    # state = dropout_letm_cell.zero_state(batch_size)
    outputs, _ = tf.nn.dynamic_rnn(dropout_letm_cell, inputs, dtype=tf.float32)
    # outputs shape: (None, sequence_length, hidden_size)
    if out == "LAST":
        # 根据提供的mask计算得到序列的实际有效长度，然后用该长度生成一个one-hot的向量，和序列输出做点乘就行了
        valid_length = tf.reduce_sum(mask, axis=-1)  # (None, )
        valid_length = valid_length - tf.constant(1.0, dtype=tf.float32)
        valid_length = tf.cast(valid_length, tf.int32)
        one_hot_length = tf.one_hot(valid_length, axis=-1, depth=sequence_length)    # (None, sequence_length)的one-hot编码

        final_output = tf.reduce_sum(outputs * tf.reshape(one_hot_length, [-1, sequence_length, 1]), axis=1)
        # (None, hidden_size)
    else:
        # average的写法有问题，先加权求和然后除以序列长度，如果长度出现0就全是nan了
        # 要先对mask向量做softmax，这样的话即使mask向量全是零也不会报错
        sequence_weight = tf.nn.softmax((mask + 5.0) * mask)    # (None, sequence_length)
        final_output = tf.reduce_sum(outputs * tf.reshape(sequence_weight, [-1, sequence_length, 1]), axis=1)
        # (None, hidden_size)
    return outputs, final_output    # 同时返回序列输出和最终的输出，序列输出在其他的模型当中需要用到

class WSAN(object):
    def __init__(self, config_):
        self.max_sentence_length1 = config_['max_sentence_length1']
        self.max_sentence_length2 = config_['max_sentence_length2']
        self.emb_dim = word_embedding_dim = config_["emb_dim"]
        self.lstm_dim = config_['lstm_dim']
        self.voc_size = config_["voc_size"]
        ave_mode = config_["ave_mode"]      # WSAN
        nb_classes = config_["nb_classes"]


        # define the input of the models
        self.x1 = tf.placeholder(dtype=tf.int32, shape=(None, self.max_sentence_length1), name="sentence_1")
        self.x2 = tf.placeholder(dtype=tf.int32, shape=(None, self.max_sentence_length2), name="sentence_2")
        self.x_mask1 = tf.placeholder(dtype=tf.float32, shape=(None, self.max_sentence_length1), name="mask_1")
        self.x_mask2 = tf.placeholder(dtype=tf.float32, shape=(None, self.max_sentence_length2), name="mask_2")
        self.y = tf.placeholder(dtype=tf.int32, shape=(None, ), name="label")
        self.keep_prob = tf.placeholder(dtype=tf.float32)
        self.word_keep_prob = tf.placeholder(dtype=tf.float32)

        # define learning rate
        self.learning_rate = tf.get_variable("lr", dtype=tf.float32, initializer=config_["lr"], trainable=False)
        self.new_lr = tf.placeholder(dtype=tf.float32)
        self.assign_lr_op = tf.assign(self.learning_rate, self.new_lr)

        # embedding mapping
        with tf.device("/cpu:0"):
            if "word_emb" in config:
                self.word_lookup_table = tf.get_variable("word_lookup", initializer=config_["word_emb"], dtype=tf.float32)
            else:
                self.word_lookup_table = tf.get_variable("word_lookup", shape=(self.voc_size, self.emb_dim), dtype=tf.float32)
            self.word_embedding_1 = tf.nn.embedding_lookup(self.word_lookup_table, self.x1)
            self.word_embedding_2 = tf.nn.embedding_lookup(self.word_lookup_table, self.x2)

        # dropout
        self.word_embedding_1 = tf.nn.dropout(self.word_embedding_1, self.keep_prob)
        self.word_embedding_2 = tf.nn.dropout(self.word_embedding_2, self.keep_prob)

        # word dropout
        self.word_embedding_1 = self.word_dropout_layer(self.word_embedding_1, self.x_mask1, self.word_keep_prob)
        self.word_embedding_2 = self.word_dropout_layer(self.word_embedding_2, self.x_mask2, self.word_keep_prob)

        # if use preprocess before weighting
        if config_["preprocess"] == "lstm":
            with tf.variable_scope("lstm", reuse=tf.AUTO_REUSE):
                self.word_embedding_1, _ = my_lstm_layer(self.word_embedding_1, self.lstm_dim, self.word_keep_prob, self.max_sentence_length1, mask=self.x_mask1)
    
                self.word_embedding_2, _ = my_lstm_layer(self.word_embedding_2, self.lstm_dim, self.word_keep_prob, self.max_sentence_length2, mask=self.x_mask2)
            self.emb_dim = self.lstm_dim

    
        # compute average representation according to the ave mode
        if ave_mode == "ave":
            self.softmax_mask1 = tf.nn.softmax(self.x_mask1 * 10.0)
            self.softmax_mask2 = tf.nn.softmax(self.x_mask2 * 10.0)
            # shape: (None, max_sentence_length)
            self.sentence_pre1 = tf.reduce_sum(self.word_embedding_1 * tf.expand_dims(self.softmax_mask1, -1), axis=-2)
            self.sentence_pre2 = tf.reduce_sum(self.word_embedding_2 * tf.expand_dims(self.softmax_mask2, -1), axis=-2)

        if ave_mode == "wsan":
            with tf.variable_scope("wsan", reuse=tf.AUTO_REUSE):
                self.attention_logits1, self.attention_weights1, self.sentence_pre1 =\
                    self.windows_sa_layer(self.word_embedding_1,
                                          self.x_mask1,
                                          [5, self.emb_dim, 1, 100],
                                          # extra_guider=[self.guider, char_embedding_dim],
                                          activation=tf.nn.relu)

                self.attention_logits2, self.attention_weights2, self.sentence_pre2 =\
                    self.windows_sa_layer(self.word_embedding_2,
                                          self.x_mask2,
                                          [5, self.emb_dim, 1, 100],
                                          # extra_guider=[self.guider, char_embedding_dim],
                                          activation=tf.nn.relu)

        if ave_mode == "gated-window-sa":
            with tf.variable_scope("gated-window-sa", reuse=tf.AUTO_REUSE):
                self.attention_logits1, self.attention_weights1, self.sentence_pre1 = \
                    self.gated_window_sa_layer(self.word_embedding_1,
                                               self.x_mask1,
                                               [5, self.emb_dim, 1, self.emb_dim],
                                               # extra_guider=[self.guider, char_embedding_dim],
                                               activation=tf.nn.relu)

                self.attention_logits2, self.attention_weights2, self.sentence_pre2 = \
                    self.gated_window_sa_layer(self.word_embedding_2,
                                               self.x_mask2,
                                               [5, self.emb_dim, 1, self.emb_dim],
                                               # extra_guider=[self.guider, char_embedding_dim],
                                               activation=tf.nn.relu)

        # following the general operator, concat dot and delta
        self.final_features = tf.concat([self.sentence_pre1*self.sentence_pre2, tf.abs(self.sentence_pre1 - self.sentence_pre2)], axis=-1)

        self.dense_layer_1 = self.dense_layer(self.final_features, [2*self.emb_dim, 50], activation=tf.nn.relu, prefix="dense_layer_1")
        self.dense_layer_2 = self.dense_layer(self.dense_layer_1[-1], [50, nb_classes], activation=identical,  prefix="dense_layer_2")
        self.logits = self.dense_layer_2[-1]
        self.prob = tf.nn.softmax(self.logits)


        # for SNLI dataset, cross entropy is used as loss
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
        self.loss = tf.reduce_mean(self.loss)

        vars = tf.trainable_variables()
        logger.info("Trainable parameters: \n")
        for v in vars:
            logger.info(v.name)
        
        # self.lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * 3e-5
        # self.loss = self.loss + self.lossL2
        # tf.add_to_collection("losses", self.loss)ss

        #self.cost = tf.add_n(tf.get_collection("losses"))ss
        # self.cost = tf.reduce_mean(self.cost)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        # self.train_op = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        # self.train_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # 定义正确的样本数，用来统计模型的正确率s
        self.prediction = tf.arg_max(self.logits, dimension=-1, output_type=tf.int32)
        self.tmp = tf.cast(tf.equal(self.prediction, self.y), tf.int32)
        self.correct_num = tf.reduce_sum(self.tmp)


    def dense_layer(self, inputs, weights_size, activation=identical, prefix="dense", regularization=False, dropout=False):
        '''
        :param inputs: dense层的输入
        :param weights_size: 权重的形状
        :param activation: 激活函数
        :param prefix: 前缀
        :return: 经过全连接层计算之后的结果
        '''
        if dropout:
            inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob)
        weights = xavier_weights_variable(weights_size, prefix)
        bias = bias_variable([weights_size[-1], ], prefix)
        outputs = activation(tf.add(tf.matmul(inputs, weights), bias))
        if regularization:
            tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.01)(weights))     # 添加L2正则化项

        return weights, bias, outputs

  
    def windows_sa_layer(self, inputs, mask, kernel_size, extra_guider=None, activation=identical, prefix="window-sa-layer"):
        '''
        用卷积的形式来计算以一个窗口的单词作为guider的SA layer，返回的是经过SA层之后加权平均的结果，即最终的句子表示
        :param inputs: 输入序列 (nb_samples, sentence_length, dim_word)
        :param mask: mask矩阵 (nb_samples, sentence_length)
        :param kernel_size: 卷积核的大小 (window_size, dim_word, 1, hidden_size)
        :param activation: 激活函数
        :param extra_guider: 额外的SA的输入 [extra_guider, dim_guider] or None/shape: (None, sentence_length, dim_guider)
        :param prefix: 前缀
        :return:
        '''
        nb_samples = tf.shape(inputs)[0]
        dim_word = tf.shape(inputs)[-1]
        sentence_length = tf.shape(inputs)[1]
        # 手动padding，只在某一个维度上进行padding，不知道这个能不能stride实现，这个后面再看
        nb_pad = int((kernel_size[0]-1)/2)
        padded_inputs = tf.concat([tf.zeros([nb_samples, 1, dim_word], dtype=tf.float32), ]*nb_pad
                                   + [inputs, ]
                                   + [tf.zeros([nb_samples, 1, dim_word], dtype=tf.float32), ]*nb_pad, axis=1)

        inputs_ = tf.expand_dims(padded_inputs, axis=-1)
        # (nb_samples, sentence_length+4, dim_word, 1)

        conv_w = xavier_weights_variable(kernel_size, prefix+"-conv")
        conv_b = bias_variable([kernel_size[-1], ], prefix+"-conv")

        hidden = activation(tf.add(tf.nn.conv2d(inputs_, conv_w, strides=(1, 1, 1, 1), padding="VALID"), conv_b))
        # (nb_samples, sentence_length, 1, hidden_size)

        hidden = tf.reshape(hidden, [nb_samples, sentence_length, kernel_size[-1]])
        # (nb_samples, sentence_length, hidden_size)

        hidden = tf.reshape(hidden, [-1, kernel_size[-1]])
        # (nb_samples*sentenec_length, hidden_size)

        dense_w = xavier_weights_variable([kernel_size[-1], 1], prefix + "-dense")

        # if extra_guider is not None:
        #     hidden = tf.concat([hidden, extra_guider[0]], axis=-1)
        #     # (None, sentence_length, hidden_size+dim_guider)
        #
        #     dense_w = xavier_weights_variable([kernel_size[-1]+extra_guider[1], 1], prefix+"-dense")
        # else:
        #     dense_w = xavier_weights_variable([kernel_size[-1], 1], prefix+"-dense")

        logits = tf.reshape(tf.matmul(hidden, dense_w) * 2.0, [-1, sentence_length])
        # shape: (nb_samples, sentence_length)

        # self_weights = self.margin_based_softmax(logits, mask, margin=0.1)
        self_weights = tf.nn.softmax((logits+10.0) * mask)

        # penalty_to_self_weights(self_weights, alpha=0.01)
        sentence_pre = tf.reduce_sum(tf.expand_dims(self_weights, axis=-1) * inputs, axis=1)    # nb_samples, dim_word

        return logits, self_weights, sentence_pre

    def gated_window_sa_layer(self, inputs, mask, kernel_size, extra_guider=None, activation=identical, prefix="gated-window-sa-layer"):
        '''
        基于门的窗口化的self-attention的计算方法，这个方法和普通的window-sa的区别在于，计算的权重是针对每一个维度的
        前面的计算操作都是一样的
        :param inputs:
        :param mask:
        :param kernel_size:
        :param extra_guider:
        :param activation:
        :param prefix:
        :return:
        '''
        nb_samples = tf.shape(inputs)[0]
        dim_word = tf.shape(inputs)[-1]
        sentence_length = tf.shape(inputs)[1]
        # 手动padding，只在某一个维度上进行padding，不知道这个能不能stride实现，这个后面再看
        nb_pad = int((kernel_size[0] - 1) / 2)
        padded_inputs = tf.concat([tf.zeros([nb_samples, 1, dim_word], dtype=tf.float32), ] * nb_pad
                                  + [inputs, ]
                                  + [tf.zeros([nb_samples, 1, dim_word], dtype=tf.float32), ] * nb_pad, axis=1)

        inputs_ = tf.expand_dims(padded_inputs, axis=-1)
        # (nb_samples, sentence_length+2*nb_pad, dim_word, 1)

        conv_w = xavier_weights_variable(kernel_size, prefix + "-conv")
        conv_b = bias_variable([kernel_size[-1], ], prefix + "-conv")

        hidden = activation(tf.add(tf.nn.conv2d(inputs_, conv_w, strides=(1, 1, 1, 1), padding="VALID"), conv_b) / 5.0) * 5.0
        # (nb_samples, sentence_length, 1, hidden_size)   scaled tanh is used to reduce the number of parameters

        logits = tf.reshape(hidden, [nb_samples, sentence_length,  kernel_size[-1]])
        # (nb_samples, sentenec_length, hidden_size)

        prob = tf.nn.softmax((logits+10.0)*tf.expand_dims(mask, axis=-1), dim=1)     # 对于每一个维度，都在整个序列上进行softmax的计算
        # (nb_samples, sentenec_length, hidden_size) 得到的feature-wise的权重
        # prob = self.margin_based_softmax(logits, mask, rank=3)

        sentence_pre = tf.reduce_sum(prob * inputs, axis=1)     # (nb_samples, dim_word)

        # tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(0.001)(conv_w))     # 添加L2正则化项

        return logits, prob, sentence_pre

    def word_dropout_layer(self, inputs, mask, keep_prob):
        '''
        word dropout层，针对输入以单词为base进行drop,普通的dropout是针对每一维来进行的
        :param inputs: (None, sentence_length, dim_pre)
        :param mask: (None, sentence_length)
        :param keep_prob:
        :return:
        '''
        dropouted_mask = tf.nn.dropout(mask, keep_prob=keep_prob)
        outputs = tf.expand_dims(dropouted_mask, axis=-1) * inputs
        
        return outputs
  

    def assign_lr(self, session, new_lr):
        '''
        调整模型的学习速率
        :param session: 所在的会话
        :param new_lr: 新的学习速率
        :return:
        '''
        session.run(self.assign_lr_op, feed_dict={self.new_lr: new_lr})

    def assign_batch_size(self, session, new_batch_size):
        session.run(self.assign_bs_op, feed_dict={self.new_batch_size: new_batch_size})


if __name__ == "__main__":
    config = utils.load_config_from_file("config", "snli")
    logger.info("build model ...")
    wsan = WSAN(config)
    logger.info("build model successfully")
    

