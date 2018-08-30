# coding: utf-8
import time
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pickle
import math
import os
import sys
# from DataUtils import get_input_mask, display_weights, getTestDataFromSubj
# from preprocess_data import charLevelMR, char_lelelSUBJ,
# char_levelTREC, SST, get_specific_emb, CR, SST2
# from tag_attention import getSubjCaseInputs
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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


def word_level_cnn(inputs, sentence_length, kernel_size_list, keep_prob, activation=tf.nn.relu):
    '''
    输入为单词序列表示，即输入图，进行卷积的计算，然后得到结果 (文本卷积一般都会采用1-d卷积)
    :param inputs: 输入图 shape: (nb_samples, sentence_length, dim_word_pre, 1)
    :param sentence_length: 句子长度
    :param kernel_size_list: 卷积核的形状(是一个list，表示不同形状的多肽卷积核)
    :param keep_prob: 用于CNN层的dropout
    :param activation: 激活函数
    :return: 返回经过卷积操作和global_pooling的结果,以及参数w/b
    '''
    # [3/4/5, 300, 1, 100]
    # 常见的文本卷积的kernel为: (3, 4, 5) * 100 然后global_pooling
    inputs = tf.nn.dropout(inputs, keep_prob)
    features = []
    w = []
    b = []
    feature_size = 0
    for kernel_size in kernel_size_list:
        print("kernel size is: ", kernel_size)
        feature_size += kernel_size[-1]
        conv_w = xavier_weights_variable(shape=kernel_size, name_prefix="sentence_cnn_" + str(kernel_size[0]))
        conv_b = bias_variable(shape=[kernel_size[-1], ], name_prefix="sentence_cnn_" + str(kernel_size[0]))
        conv_out = tf.nn.conv2d(inputs, conv_w, padding="VALID", strides=[1, 1, 1, 1])
        conv_out = activation(conv_out + conv_b)
        # 在序列长度这一个维度上进行max pooling
        max_pooling_out = tf.nn.max_pool(
            conv_out, ksize=[1, sentence_length - kernel_size[0] + 1, 1, 1], padding="VALID",
            strides=[1, 1, 1, 1])
        w.append(conv_w)
        b.append(conv_b)
        features.append(max_pooling_out)    # shape: (nb_samples, 1, 1, 100)
        # outputs = tf.nn.dropout(max_pooling_out, keep_prob=keep_prob)
        # return conv_w, conv_b, outputs
    print("feature_size is: ", feature_size)
    sentence_pre = tf.reshape(tf.concat(features, axis=-1), [-1, feature_size])
    # shape: (nb_samples, 1, 1, 300) --> (nb_samples, 300)
    return w, b, sentence_pre


class WSAN(object):
    def __init__(self, config_):
        # self.config = config_
        self.max_sentence_length1 = config_['max_sentence_length1']
        self.max_sentence_length2 = config_['max_sentence_length2']
        self.emb_dim = word_embedding_dim = config_["emb_dim"]
        self.lstm_dim = config_['lstm_dim']
        self.voc_size = config_["voc_size"]
        ave_mode = config_["ave_mode"]      # "SA" or "window-sa"
        nb_classes = config_["nb_classes"]

        # 类别标签
        # self.y = tf.placeholder(tf.int64, (None, ))

        # 相似度向量标签
        self.score_vec = tf.placeholder(tf.float32, shape=(None, nb_classes), name="score_vector")

    
        self.x1 = tf.placeholder(dtype=tf.int32, shape=(None, self.max_sentence_length1), name="sentence_1")
        self.x2 = tf.placeholder(dtype=tf.int32, shape=(None, self.max_sentence_length2), name="sentence_2")
        self.x_mask1 = tf.placeholder(dtype=tf.float32, shape=(None, self.max_sentence_length1), name="mask_1")
        self.x_mask2 = tf.placeholder(dtype=tf.float32, shape=(None, self.max_sentence_length2), name="mask_2")
        self.keep_prob = tf.placeholder(dtype=tf.float32)
        self.word_keep_prob = tf.placeholder(dtype=tf.float32)

        self.learning_rate = tf.get_variable("lr", dtype=tf.float32, initializer=config_["lr"], trainable=False)
        self.new_lr = tf.placeholder(dtype=tf.float32)
        self.assign_lr_op = tf.assign(self.learning_rate, self.new_lr)

        # self.batch_size = tf.get_variable("bs", dtype=tf.int32, initializer=config_["batch_size"], trainable=False)
        # self.new_batch_size = tf.placeholder(dtype=tf.int32)
        # self.assign_bs_op = tf.assign(self.batch_size, self.new_batch_size)
        # self.batch_size = 32

        # 如果加入word-level的词向量的话
        with tf.device("/cpu:0"):
            if config_["word_emb"] is not None:
                self.word_lookup_table = tf.get_variable("word_lookup", initializer=config_["word_emb"], dtype=tf.float32, trainable=False)
            else:
                self.word_lookup_table = tf.get_variable("word_lookup", [self.voc_size, self.emb_dim], dtype=tf.float32)
            self.word_embedding_1 = tf.nn.embedding_lookup(self.word_lookup_table, self.x1)
            self.word_embedding_2 = tf.nn.embedding_lookup(self.word_lookup_table, self.x2)
            # shape: (None, max_sentence_length1, emb_dim)/(None, max_sentence_length2, emb_dim)

        # 普通的dropout
        self.word_embedding_1 = tf.nn.dropout(self.word_embedding_1, self.keep_prob)
        self.word_embedding_2 = tf.nn.dropout(self.word_embedding_2, self.keep_prob)

        # 单词级别的dropout
        # self.word_embedding_1 = self.word_dropout_layer(self.word_embedding_1, self.x_mask1, self.word_keep_prob)
        # self.word_embedding_2 = self.word_dropout_layer(self.word_embedding_2, self.x_mask2, self.word_keep_prob)

        # word embedding后面跟一个全连接层
        # self.word_representation = self.dense_layer_t3(self.word_representation,
        #                                                max_sentence_length,
        #                                                [self.word_representation_dim, self.word_representation_dim],
        #                                                tf.nn.relu,
        #                                                prefix="word2hidden-dense-layer")[-1]

        if config_["preprocess"] == "lstm":
            with tf.variable_scope("lstm", reuse=tf.AUTO_REUSE):
                self.word_embedding_1, _ = my_lstm_layer(self.word_embedding_1, self.lstm_dim, self.word_keep_prob, self.max_sentence_length1, mask=self.x_mask1)
    
                self.word_embedding_2, _ = my_lstm_layer(self.word_embedding_2, self.lstm_dim, self.word_keep_prob, self.max_sentence_length2, mask=self.x_mask2)
            self.emb_dim = self.lstm_dim

        if config_["preprocess"] == "blstm":
            self.word_representation = word_level_blstm(self.word_representation,
                                                        max_sentence_length,
                                                        150,
                                                        keep_prob=1.0,
                                                        batch_size=self.batch_size)


        if ave_mode == "ave":
            self.softmax_mask1 = tf.nn.softmax(self.x_mask1 * 10.0)
            self.softmax_mask2 = tf.nn.softmax(self.x_mask2 * 10.0)
            # shape: (None, max_sentence_length)
            self.sentence_pre1 = tf.reduce_sum(self.word_embedding_1 * tf.expand_dims(self.softmax_mask1, -1), axis=-2)
            self.sentence_pre2 = tf.reduce_sum(self.word_embedding_2 * tf.expand_dims(self.softmax_mask2, -1), axis=-2)
        if ave_mode == "sa":
            self.sentence_pre \
                = self.intra_weighting_layer(self.word_representation, guider=self.guider,
                                             guider_size=char_embedding_dim, hidden_size=config_["intra_hidden_size"],
                                             mask=self.x_word_mask, activation=tf.nn.relu)[-1]
            # self.word_representation = self.intra_weighting_layer_without_sum(self.word_representation,
            #                                                                   self.guider,
            #                                                                   word_embedding_dim,
            #                                                                   self.config["intra_hidden_size"])[-1]

        if ave_mode == "window-sa":
            with tf.variable_scope("window-sa", reuse=tf.AUTO_REUSE):
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


        '''
        if "cnn" in mode:
            cnn_config = self.config["word-level-cnn-config"]
            self.word_representation = tf.reshape(self.word_representation,
                                                  [-1, max_sentence_length, self.word_representation_dim, 1])
            self.sentence_pre = word_level_cnn(self.word_representation,
                                               max_sentence_length,
                                               cnn_config["kernel_size_list"],
                                               1.0,
                                               tf.nn.relu)[-1]


        if "lstm" in mode:
            lstm_config = self.config["word-level-lstm-config"]
            # print(tf.shape(self.word_representation))
            self.sentence_pre = word_level_lstm(self.word_representation,
                                                self.x_word_mask,
                                                max_sentence_length,
                                                lstm_config["dim_hidden_list"],
                                                lstm_config["keep_prob"],
                                                self.batch_size)
        '''


        # self.sentence_pre1 = tf.nn.dropout(self.sentence_pre1, self.keep_prob)
        # self.sentence_pre2 = tf.nn.dropout(self.sentence_pre2, self.keep_prob)
        self.final_features = tf.concat([self.sentence_pre1*self.sentence_pre2, tf.abs(self.sentence_pre1 - self.sentence_pre2)], axis=-1)
        # shape: (None, 4*emb_dim)

        self.dense_layer_1 = self.dense_layer(self.final_features, [2*self.lstm_dim, 50], activation=tf.nn.relu, prefix="dense_layer_1", dropout=False)
        self.dense_layer_2 = self.dense_layer(self.dense_layer_1[-1], [50, nb_classes], activation=identical,  prefix="dense_layer_2")
        self.logits = self.dense_layer_2[-1]
        self.prob = tf.nn.softmax(self.logits)

        # 调用这个函数的时候，必需要带上参数的名字
        # “Only call `sparse_softmax_cross_entropy_with_logits` with named arguments (labels=..., logits=..., ...)
        # self.cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.y, [-1, ]), logits=self.dense_layer_2[-1])
        # todo: 自己写cross-entropy,check一下是不是sparse_softmax_cross_entropy_with_logits的问题
        # ps: 不是上面的函数的问题
        # self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)

        # SICK relatdeness任务，label是一个5-d vector, 非one-hot 编码
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.score_vec, logits=self.logits)
        self.loss = tf.reduce_mean(self.loss)

        vars = tf.trainable_variables()
        self.lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * 3e-5

        self.loss = self.loss + self.lossL2
        # tf.add_to_collection("losses", self.loss)ss

        #self.cost = tf.add_n(tf.get_collection("losses"))ss
        # self.cost = tf.reduce_mean(self.cost)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        # self.train_op = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        # self.train_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # 定义正确的样本数，用来统计模型的正确率s
        # self.prediction = tf.arg_max(self.logits, dimension=-1)
        # self.prediction = tf.assert_type(self.prediction, tf.int32)
        # self.tmp = tf.cast(tf.equal(self.prediction, self.y), tf.int32)
        # self.correct_num = tf.reduce_sum(self.tmp)


    def conv_and_global_pooling(self, input_map, kernels, prefix="conv"):
        '''
        :param input_map: 输入特征图(或者原始输入)
        :param kernels: 卷积核的数量和大小(height, width, in_channel, nb_kernel)
        :param prefix: 前缀，用来规范参数的名字
        :return: 经过1D卷积操作和max pooling的结果(即完成一个卷积+max pooling的计算)
        '''
        conv_w = weight_variable(shape=kernels, name_prefix=prefix)
        conv_b = bias_variable(shape=[kernels[-1], ], name_prefix=prefix)
        conv_out = tf.nn.conv2d(input_map, conv_w, padding="VALID", strides=[1, 1, 1, 1])
        conv_out = tf.nn.relu(conv_out + conv_b)
        # 在序列长度这一个维度上进行max pooling
        max_pooling_out = tf.nn.max_pool(
            conv_out, ksize=[1, self.config["max_char_len"]-kernels[0]+1, 1, 1], padding="VALID", strides=[1, 1, 1, 1])

        return conv_w, conv_b, max_pooling_out

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

    def dense_layer_t3(self, inputs, sentence_length, weights_size, activation=identical, prefix="dense"):
        inputs = tf.reshape(inputs, [-1, weights_size[0]])
        weights = xavier_weights_variable(weights_size, prefix)
        bias = bias_variable([weights_size[-1], ], prefix)
        outputs = activation(tf.add(tf.matmul(inputs, weights), bias))

        return weights, bias, tf.reshape(outputs, [-1, sentence_length, weights_size[-1]])

    def intra_weighting_layer(self, inputs, guider, guider_size, hidden_size, mask, activation=identical, prefix="intra-weight"):
        '''
        :param inputs: self-attention层的输入,这个模型当中是三维的输入(nb_samples, sentence_length, dim_pre)
        :param guider_size: 用于attention的引导信息的维度 (nb_samples, sentence_length, guider_size)
        :param guider: 引导信息
        :param hidden_size: 隐层大小
        :param activation: 激活函数
        :param mask: mask矩阵 shape: (nb_samples, sentence_length)
        :param prefix: 前缀
        :return:
        '''
        # in_size = tf.shape(inputs)[-1]
        weights_h = xavier_weights_variable([guider_size, hidden_size], prefix+"-h")
        bias_h = bias_variable([hidden_size, ], prefix+"-h")
        guider_ = tf.reshape(guider, [-1, guider_size])   # 铺开到2维，成为一个矩阵
        hidden_units = tf.matmul(guider_, weights_h) + bias_h
        hidden_units = activation(hidden_units)   # (nb_samples*sentence_length, hidden_size)
        weights_o = xavier_weights_variable([hidden_size, 1], prefix+"-o")

        self_logits = tf.matmul(hidden_units, weights_o) # (nb_samples*sentence_length, 1)
        self_logits = tf.reshape(self_logits, tf.shape(inputs)[:-1])    # (nb_samples, sentence_length)

        # 在softmax层之前，需要进行mask的操作
        self_logits = (self_logits + 5.0) * mask
        self_weights = tf.nn.softmax(self_logits)   # 归一化之后的 (nb_samples, sentence_length)

        return [weights_h, weights_o], [bias_h], tf.reduce_sum(inputs * tf.expand_dims(self_weights, axis=-1), axis=-2)

    def intra_weighting_layer_without_sum(self, inputs, guider, guider_size, hidden_size, activation=identical, prefix="intra-weight-no-sum"):
        '''
        不做加权平均的SA-layer,只是对序列做一个加权，返回还是同样形状的序列。 这个部分不需要MASK，因为padding部分的权重，会在后面的mask进行处理
        ps: 为了方便接口一致，输出改为计算得到的logits(没有经过softmax函数)
        :param inputs: 输入 (nb_samples, sentence_length, dim_pre)
        :param guider: 计算SA的guider (nb_samples, sentence_length, guider_size)
        :param guider_size: guider的维度
        :param hidden_size: 隐层维度
        :param activation: 激活函数
        :param prefix: 前缀
        :return:
        '''
        weight_h = xavier_weights_variable([guider_size, hidden_size], prefix+"-h")
        bias_h = bias_variable([hidden_size, ], prefix+"-h")
        weight_o = xavier_weights_variable([hidden_size, 1], prefix+"-o")
        bias_o = bias_variable([1, ], prefix+"-o")

        guider_ = tf.reshape(guider, [-1, guider_size])
        hidden_units = activation(tf.matmul(guider_, weight_h) + bias_h)
        self_logits = activation(tf.matmul(hidden_units, weight_o) + bias_o)    # shape: (nb_samples*sentence_length, 1)
        self_logits = tf.reshape(self_logits, tf.shape(inputs)[:-1])    # (nb_samples, sentence_length)

        # intra_weighted_pre = tf.expand_dims(self_logits, axis=-1) * inputs   # (nb_samples, sentence_length, dim_pre)

        return [weight_h, weight_o], [bias_h, bias_o], self_logits

    def semantic_projection_layer(self, input1, input2, input1_dim, input2_dim, sentence_length, proj_dim, activation=identical, prefix="semantic-projection"):
        '''
        语义投影层的定义，输入有两部分，分别是来自于字符级别的单词表示和单词级别的单词表示
        (nb_samples, sentence_length, input1_dim), (nb_samples, sentence_length, input2_dim)
            --> (nb_samples, sentence_length, proj_dim)
        :param input1: 输入1(char-level)
        :param input2: 输入2(word-level)
        :param input1_dim: 输入1的维度
        :param input2_dim: 输入2的维度
        :param sentence_length: 句子长度
        :param proj_dim: 投影层的维度
        :param activation: 激活函数
        :param prefix: 命名前缀
        :return:
        '''
        # original_shape = tf.shape(input1)
        input1_ = tf.reshape(input1, [-1, input1_dim])   # (nb_samples*sentence_length, input1_dim)
        input2_ = tf.reshape(input2, [-1, input2_dim])   # (nb_samples*sentence_length, input2_dim)
        proj_w_1 = xavier_weights_variable([input1_dim, proj_dim], name_prefix=prefix+"-char2proj")
        proj_w_2 = xavier_weights_variable([input2_dim, proj_dim], name_prefix=prefix+"-word2proj")
        proj_b = bias_variable([proj_dim, ], name_prefix=prefix)
        outputs = tf.add(tf.matmul(input1_, proj_w_1), tf.matmul(input2_, proj_w_2))
        outputs = activation(outputs + proj_b)
        # shape: (nb_samples*sentence_length, proj_dim)

        return [proj_w_1, proj_w_2, proj_b], tf.reshape(outputs, [-1, sentence_length, proj_dim])

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

    def multi_window_max_pooling_sa_layer(self, inputs, mask, kernel_sizes, extra_guider=None, activation=identical, prefix="multi-window-max-pooling-sa-layer"):
        '''
        与 window-sa差不多，但是有不同长度的window-sa，然后对不同长度的结果进行max_pooling,获取单次最有可能的n-gram信息
        参数同window-sa layer
        :param inputs:
        :param mask:
        :param kernel_sizes:
        :param extra_guider
        :param activation:
        :param prefix:
        :return:
        '''

        # 用一个1-D卷积的操作来实现window-sa
        def conv_for_window_sa(inputs, kernel_size, nb_left_pad, nb_right_pad):
            nb_samples = tf.shape(inputs)[0]
            dim_word = tf.shape(inputs)[-1]
            padded_inputs = tf.concat([tf.zeros([nb_samples, 1, dim_word], dtype=tf.float32), ]*nb_left_pad
                                      + [inputs, ]
                                      + [tf.zeros([nb_samples, 1, dim_word], dtype=tf.float32), ]*nb_right_pad, axis=1)
            # (None, sentence_length+nb_left_pad+nb_right_pad, dim_word)

            inputs_ = tf.expand_dims(padded_inputs, axis=-1)
            # (nb_samples, sentence_length+nb_left_pad+nb_right_pad, dim_word, 1)

            conv_w = xavier_weights_variable(kernel_size, prefix + "-conv-{0}-{1}".format(nb_left_pad, nb_right_pad))
            conv_b = bias_variable([kernel_size[-1], ], prefix + "-conv-{0}-{1}".format(nb_left_pad, nb_right_pad))

            hidden = activation(tf.add(tf.nn.conv2d(inputs_, conv_w, strides=(1, 1, 1, 1), padding="VALID"), conv_b))
            # (nb_samples, sentence_length, 1, hidden_size)

            hidden = tf.reshape(hidden, [-1, kernel_size[-1]])
            # (nb_samples*sentence_length, hidden_size)

            return hidden

        hidden_list = []     # list of (nb_samples*sentence_length, hidden_size)
        for kernel_size in kernel_sizes:
            window_size = kernel_size[0]
            if window_size % 2 == 1:
                hidden_list.append(conv_for_window_sa(inputs, kernel_size, (window_size-1)/2, (window_size-1)/2))
            elif window_size == 2:
                hidden_list.append(conv_for_window_sa(inputs, kernel_size, 0, 1))
                hidden_list.append(conv_for_window_sa(inputs, kernel_size, 1, 0))

        nb_samples = tf.shape(inputs)[0]
        sentence_length = tf.shape(inputs)[1]
        hidden_size = kernel_sizes[0][-1]
        multi_hidden = tf.stack(hidden_list, axis=1)    # (nb_samples*sentence_length, nb_windows_type, hidden_size)
        final_hidden = tf.reduce_max(multi_hidden, axis=-2)     # (nb_samples*sentence_length, hidden_size)

        dense_w = xavier_weights_variable([hidden_size, 1], prefix + "-dense")

        logits = tf.reshape(tf.matmul(final_hidden, dense_w) * 2.0, [-1, sentence_length])
        # shape: (nb_samples, sentence_length)

        self_weights = tf.nn.softmax((logits + 5.0) * mask)
        sentence_pre = tf.reduce_sum(tf.expand_dims(self_weights, axis=-1) * inputs, axis=1)  # nb_samples, dim_word

        return logits, self_weights, sentence_pre

    def multi_head_window_sa_layer(self, inputs, mask, kernel_size, extra_guider=None, activation=identical, nb_head=3, keep_prob=1.0, prefix="multi-haed-window-sa-layer"):
        '''
        多路的window-sa layer, 多套权重分布意味着得到了多个关注点不同的句子表示
        除了nb_head，其他的参数同window-sa layer一致
        :param inputs:
        :param mask:
        :param kernel_size:
        :param extra_guider: 额外的SA的输入 [extra_guider, dim_guider] or None
        :param activation:
        :param nb_head: 权重套数
        :param keep_prob
        :param prefix:
        :return:
        '''
        dim_word_pre = self.word_embedding_dim
        outputs = []
        w = xavier_weights_variable([dim_word_pre * nb_head, dim_word_pre], name_prefix=prefix)
        b = bias_variable([dim_word_pre, ], name_prefix=prefix)
        for head in range(nb_head):
            sentence_pre = self.windows_sa_layer(inputs, mask, kernel_size, extra_guider, activation, prefix+"-{0}".format(head+1))[-1]
            outputs.append(sentence_pre)    # list of (None, dim_word_pre)
        sentence_pre = tf.concat(outputs, axis=-1)      # (None, dim_word_pre*nb_head)
        sentence_pre = tf.nn.dropout(sentence_pre, keep_prob=keep_prob)
        sentence_pre = activation(tf.matmul(sentence_pre, w) + b)   # (None, dim_word_pre)

        # 后面用一个 Highway Network来提取特征
        w_t_highway = xavier_weights_variable([self.word_embedding_dim, self.word_embedding_dim], name_prefix=prefix+"-highway-t")
        b_t_highway = bias_variable([self.word_embedding_dim, ], name_prefix=prefix+"-highway-t")
        w_h_highway = xavier_weights_variable([self.word_embedding_dim, self.word_embedding_dim], name_prefix=prefix+"-highway-h")
        b_h_highway = bias_variable([self.word_embedding_dim, ], name_prefix=prefix+"-highway-h")

        highway_gate = tf.nn.sigmoid(tf.matmul(sentence_pre, w_t_highway) + b_t_highway)
        highway_hidden_units = activation(tf.matmul(sentence_pre, w_h_highway) + b_h_highway)

        highway_outputs = highway_gate * highway_hidden_units + (1.0 - highway_gate) * sentence_pre

        return highway_outputs

    def lstm_sa_layer(self, inputs, mask, dims_hidden_unit, dim_dense_hidden, sentence_length, keep_prob, batch_size_, extra_guider):
        '''
        lstm-sa layer 顾名思义，使用lstm的序列输出作为sa layer的输入
        :param inputs: 输入序列 (None, sentence_length, dim_word)
        :param mask: mask矩阵 (None, sentence_length)
        :param dims_hidden_unit: lstm的隐层维度 是一个list，可能是多层的lstm, 这里不包括 bi-lstm
        :param dim_dense_hidden: lstm之后的dense层的隐层维度
        :param sentence_length: 句子长度
        :param keep_prob: lstm层的 keep_prob
        :param batch_size_: batch_size
        :param extra_guider: list of 2 elements: [额外的引导信息, 引导信息的维度]
        :return:
        '''
        # 调用LSTM的接口，返回的是经过lstm层计算的序列输出  (None, sentence_length, dims_hidden_unit[-1])
        lstm_outputs = word_level_lstm(inputs, mask, sentence_length, dims_hidden_unit, keep_prob, batch_size_)

        # sa layer参数定义、计算定义
        # 调用dense_layer_t3这个接口  (None, sentence_length, dim_dense_hidden)
        hidden = self.dense_layer_t3(lstm_outputs, sentence_length, [dims_hidden_unit[-1], dim_dense_hidden],
                                     activation=tf.nn.relu, prefix="lstm-sa-dense-1")[-1]

        hidden = tf.reshape(hidden, [-1, dim_dense_hidden])
        hidden = tf.concat([hidden, extra_guider[0]], axis=-1)     # (None, sentence_length, dim_dense_hidden+dim_guider)

        dense_w = xavier_weights_variable([dim_dense_hidden+extra_guider[1], 1], "lstm-sa-dense-2")
        logits = tf.reshape(tf.matmul(hidden, dense_w), [-1, sentence_length])

        self_weights = tf.nn.softmax((logits+5.0) * mask)
        sentence_pre = tf.reduce_sum(tf.expand_dims(self_weights, axis=-1) * inputs, axis=1)    # (None, dim_word)

        return sentence_pre

    def multi_head_dot_sa_layer(self, inputs, mask, dim_pre, dim_hidden_unit, nb_head, sentence_length, keep_prob, prefix="multi-head-dot-sa"):
        '''
        多路的乘法式的SA，输入为一个序列，用每一个位置的向量跟每一个结果计算一个相似度，
        进而得到一个权重，更新该位置的向量为新的序列的加权表示
        ps: 这里的attention的方式为 (WK dot UK) / sqrt(dim_pre)
        ps: 暂时为1路，后面再试一下多路的情况
        :param nb_head: 重复的路数
        :param inputs: 输入序列 (None, sentence_length, dim_pre)
        :param mask: (None, sentence_length)
        :param dim_pre: 输入序列的表示的维度
        :param dim_hidden_unit: WK dot UQ 的维度
        :param sentence_length: 句子长度
        :param keep_prob: 1-dropout_rate, 这里的 attention dropout 作用在softmax层之后,
        :return:
        '''
        sa_outputs = []
        proj_w = []
        # 多路的SA，每一路的区别就是第一步的线性变换的参数不同，其他计算过程都是一样的
        for head in range(nb_head):
            proj_w_k = xavier_weights_variable([dim_pre, dim_hidden_unit], name_prefix=prefix+"-proj_k-"+str(head))
            proj_w_q = xavier_weights_variable([dim_pre, dim_hidden_unit], name_prefix=prefix+"-proj_q-"+str(head))
            proj_w_v = xavier_weights_variable([dim_pre, dim_hidden_unit], name_prefix=prefix+"-proj_v-"+str(head))

            # 对attention的三个输入KQV进行线性映射，在SA中KQV都是序列自身
            projected_query = self.matmul_for_rank_3(inputs, proj_w_q)     # (None, sentence_length, dim_hidden_unit)
            projected_key = self.matmul_for_rank_3(inputs, proj_w_k)
            projected_value = self.matmul_for_rank_3(inputs, proj_w_v)

            outputs = []
            for step in range(sentence_length):
                query = projected_query[:, step, :]     # (None, dim_hidden_unit)
                logits = tf.reduce_sum(tf.expand_dims(query, axis=1) * projected_key, axis=-1)   # (None, sentence_length)
                logits = logits / tf.sqrt(float(dim_pre))
                # 这一步很重要，如果没有对value处理的话值都很大，那么相对的delta就会很大，很可能attention只在某一个维度的值特别大
                sa_weights = tf.nn.softmax((logits+5.0) * mask)    # (None, sentence_length)
                sa_weights = tf.nn.dropout(sa_weights, keep_prob=keep_prob)
                weighted_value = tf.reduce_sum(tf.expand_dims(sa_weights, axis=-1)*projected_value, axis=1)
                # (None, dim_hidden_unit)
                outputs.append(weighted_value)

            # ouuputs is a list of (None, dim_hidden_unit)
            sa_output = tf.stack(outputs, axis=1)   # (None, sentence_length, dim_hidden_unit)
            sa_outputs.append(sa_output)
            proj_w += [proj_w_k, proj_w_q, proj_w_v]

        sa_outputs = tf.concat(sa_outputs, axis=-1)     # (None, sentence_length, dim_hidden_unit*nb_head)
        head_w = xavier_weights_variable([dim_hidden_unit*nb_head, dim_hidden_unit], name_prefix=prefix+"-head")
        sa_outputs = self.matmul_for_rank_3(sa_outputs, head_w)
        return proj_w, sa_outputs+inputs

    def matmul_for_rank_3(self, inputs, weights):
        '''
        由于tf自带的matmul只能支持rank-2的情况，对于rank-3的输入需要进行两次reshape
        :param inputs: rank-3的输入    (dim1, dim2, dim3)
        :param weights: rank-2的权重   (dim3, dim4)
        :return:
        '''
        input_shape = tf.shape(inputs)
        weight_shape = tf.shape(weights)
        rank_2_inputs = tf.reshape(inputs, [-1, input_shape[-1]])   # (dim1*dim2, dim3)
        rank_2_outputs = tf.matmul(rank_2_inputs, weights)      # (dim1*dim2, dim4)
        outputs = tf.reshape(rank_2_outputs, [input_shape[0], input_shape[1], weight_shape[-1]])    # (dim1, dim2, dim4)

        return outputs

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

    def gated_embedding_composition_layer(self, input_char, input_word, dim_word):
        '''
        通过门的机制对单词级别的emb和句子级别的emb进行 element-wise 结合
        返回的是结合后的每一个单词的表示 shape: (nb_samples, sentence_length, dim_word)
        :param input_char:  (nb_samples, sentence_length, dim_word)
        :param input_word:  (nb_samples, sentence_length, dim_word)
        :param dim_word:
        :return:
        '''
        w_word_gate = xavier_weights_variable([dim_word, dim_word], name_prefix="gated_embedding_composition_layer_wo")
        w_char_gate = xavier_weights_variable([dim_word, dim_word], name_prefix="gated_embedding_composition_layer_ch")
        b_gate = bias_variable([dim_word, ], name_prefix="gated_embedding_composition_layer")

        gate = tf.nn.sigmoid(self.matmul_for_rank_3(input_char, w_char_gate) +
                             self.matmul_for_rank_3(input_word, w_word_gate) +
                             b_gate)

        word_pre = gate * input_char + (1.0 - gate) * input_word

        return word_pre

    def margin_based_softmax(self, logits, mask, rank, margin=0.09):
        '''
        带阈值的softmax，将得到的概率小于阈值的概率置为0，然后再次计算softmax
        :param logits: 未经过softmax的值 (nb_samples, sentence_length)
        :param mask: 掩码 (nb_samples, sentence_length)
        :param margin: 阈值
        :param rank: 输入的tensor的rank,如果是2就是word-wise attention,如果是3就是feature-wise attention
        :return:
        '''
        if rank == 2:
            tmp_weights = tf.nn.softmax((logits + 10.0) * mask)
            tmp_weights = tf.nn.relu(tmp_weights - margin)
            tag = tf.cast(tf.not_equal(tmp_weights, 0.0), tf.float32)
            final_weights = tf.nn.softmax((tmp_weights + 10.0) * tag)
        else:
            tmp_weights = tf.nn.softmax((logits+5.0)*tf.expand_dims(mask, axis=-1), dim=1)      # 先进行一次softmax
            tmp_weights = tf.nn.relu(tmp_weights - margin)
            tag = tf.cast(tf.not_equal(tmp_weights, 0.0), tf.float32)
            final_weights = tf.nn.softmax((tmp_weights + 10.0) * tag)

        return final_weights

    def extraxt_n_gram(self, inputs, sentence_length, N):
        '''
        将(nb_samples, sentence_length, dim_word)的输入u进行n-gram的提取，得到(nb_samples, sentence_length, n*dim_word)
        :param inputs:
        :param sentence_length: 句子长度
        :param N: n-gram
        :return:
        '''
        dim_word = tf.shape(inputs)[-1]
        nb_samples = tf.shape(inputs)[0]
        nb_pad = N - 1
        inputs = tf.concat([inputs, tf.zeros([nb_samples, nb_pad, dim_word])], axis=1)

        outputs = []
        for pos in list(range(sentence_length)):
            slice_ = inputs[:, pos:pos+N, :]
            slice_ = tf.reshape(slice_, [-1, N*dim_word])
            outputs.append(slice_)
        # Now outputs is a list: sentence_length * [nb_samples, N * dim_word]

        n_gram_representation = tf.stack(outputs, axis=1)   # (nb_samples, sentence_length, N*word_dim)

        return n_gram_representation

    def n_gram_to_representation(self, n_gram_inputs, N, dim_word, prefix):
        '''
        通过self-attention，对n-gram中的各个单词的表示进行加权，得到的加权表示作为该n-gram的表示
        :param n_gram_inputs: (nb_sam, s_length, N*dim_word)
        :param N:
        :param dim_word:
        :return:
        '''
        shape_ = tf.shape(n_gram_inputs)
        nb_samples = shape_[0]
        sentence_length = shape_[1]
        w_sa = xavier_weights_variable([dim_word, 1], prefix)

        n_gram_inputs = tf.reshape(n_gram_inputs, [nb_samples, sentence_length, N, dim_word])
        n_gram_inputs = tf.reshape(n_gram_inputs, [nb_samples*sentence_length, N, dim_word])

        logits = self.matmul_for_rank_3(n_gram_inputs, w_sa)
        logits = tf.reshape(logits, [nb_samples*sentence_length, N])
        weights = tf.nn.softmax(logits)

        representation = tf.reduce_sum(tf.expand_dims(weights, axis=-1) * n_gram_inputs, axis=-2)
        # (nb_sam*s_length, dim_word)

        return w_sa, tf.reshape(representation, [nb_samples, sentence_length, dim_word])

    def combine_n_gram(self, n_grams, dim_word, prefix="n-gram-combine-layer"):
        '''
        把用不同长度的n-gram得到的表示拼接起来的层，然后用sa得到一个综合的表示
        :param n_grams: list of n-grams: [2-gram, 3-gram, 4-gram],每一个n-gram都是一个 (nb_samples, sentence_length, dim_word)
        :return:
        '''
        N = len(n_grams)
        print("N = {0}".format(N))
        multi_grams = tf.concat(n_grams, axis=-1)    # (nb_samples, sentence_length, dim_word*N)
        multi_grams = tf.nn.dropout(multi_grams, keep_prob=self.keep_prob)
        # 然后调用n_gram_to_representation函数就可以实现了
        w_combine_sa, representation = self.n_gram_to_representation(multi_grams, N, dim_word, prefix)

        return w_combine_sa, representation


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



