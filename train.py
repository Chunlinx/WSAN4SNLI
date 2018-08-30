# coding: utf-8
# author: huang ting
# created time: 2018-04-28-23:14
import tensorflow as tf
import pickle as pkl
import numpy as np
import models
import time
import os
import math
import sys
from scipy.stats import spearmanr, pearsonr
np.random.seed(1234567)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_mask(inputs):
    shape = inputs.shape
    mask = np.zeros(shape, dtype=np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            mask[i][j] = 1.0 if inputs[i,j] > 0 else 0.0

    return mask


def get_word_emb(word2idx):
    idx2word = {}
    for word, idx in word2idx.items():
        idx2word[idx] = word

    emb_path = "/home/ht/glove.6B/glove.6B.300d.txt"
    word2embedding = {}
    f = open(emb_path, 'rb')
    for line in f:
        values = line.split()
        word = values[0]
        word = word.decode()
        emb = np.asarray(values[1:], dtype='float32')
        word2embedding[word] = emb
        # print(type(word))
    f.close()
    print(len(word2embedding))
    hit_count = 0   # 统计命中数
    zero_embedding = np.zeros([300, ], dtype=np.float32)
    embedding_matrix = np.random.uniform(-0.05, 0.05, size=[len(idx2word), 300]).astype(np.float32)
    for i in range(len(word2idx)):
        if i == 0:
            emb = zero_embedding
            embedding_matrix[i] = emb
        else:
            word = idx2word[i]
            emb = word2embedding.get(word)
            if emb is not None:
                hit_count += 1
                embedding_matrix[i] = emb
    print("hit rate is {}".format(hit_count*1.0/len(word2idx)))
    print(embedding_matrix.shape)
    print(embedding_matrix[:10])
    f = open("../SICK/embedding_matrix.pkl", "wb")
    pkl.dump(embedding_matrix, f)
    f.close()


def get_pretrained_embedding(file_name, voc_size, emb_dim=300):
    f = open(file_name, 'rb')
    embs = pkl.load(f)
    f.close()
    assert embs.shape[0] == voc_size
    assert embs.shape[1] == emb_dim

    return embs


def macro_recall_rate(predictions, labels, nb_class=3):
    # 计算 macro-recall，每一类计算召回率，然后取平均;
    # 同时返回每一类的召回率和平均召回率。
    nb_per_class = np.zeros(nb_class, dtype=float)
    nb_right_per_class = np.zeros(nb_class, dtype=float)
    for p, l in zip(predictions, labels):
        nb_per_class[l] += 1
        if p == l:
            nb_right_per_class[l] += 1
    recall = nb_right_per_class / nb_per_class
    macro_recall = recall.mean()

    return recall, macro_recall


def vector2score(vectors):
    # print(len(vectors))
    # 把5-d的向量转为1-5之间的一个分数
    base = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    scores = [] 
    for vector in vectors:
        score = (base * vector).sum()
        scores.append(score)
    scores = np.array(scores, dtype=float)

    return scores


def my_spearman(scores1, scores2):
    sp =  spearmanr(scores1, scores2)
    return sp[0]


def my_pearson(scores1, scores2):
    pe =  pearsonr(scores1, scores2)
    return pe[0]



def mse(scores1, scores2):
    return np.square(scores1 - scores2).mean()


def train_epoch(session, model, batch_size, train_data, keep_prob, word_keep_prob):
    # train_data是一个dict包含了"sent1"/"sent2"/"label"

    # fetches = [model.loss, model.correct_num, model.train_op]
    fetches = [model.loss, model.prob, model.train_op]
    x1 = train_data["sent1"]
    x2 = train_data["sent2"]
    # y = train_data["label"]
    score = train_data["score"]
    # shuffle data set
    nb_samples = x1.shape[0]
    idx_list = list(range(nb_samples))
    np.random.shuffle(idx_list)
    x1 = x1[idx_list, :]
    x2 = x2[idx_list, :]
    # y = y[idx_list]
    score = score[idx_list, :]
    nb_batches = int(nb_samples * 1.0 / batch_size)
    nb_left = nb_samples - nb_batches * batch_size

    st_time = time.time()
    nb_right_sum = 0
    loss_sum = 0.0
    probs = []
    for j in range(nb_batches):
        batch_x1 = x1[j*batch_size: j*batch_size+batch_size][:]
        batch_x2 = x2[j*batch_size: j*batch_size+batch_size][:]
        batch_score = score[j*batch_size: j*batch_size+batch_size][:]
        # batch_y = y[j*batch_size: j*batch_size+batch_size]
        batch_mask1 = get_mask(batch_x1)
        batch_mask2 = get_mask(batch_x2)
        feed_dict = {model.keep_prob: keep_prob, model.word_keep_prob: word_keep_prob, model.score_vec: batch_score,
                     # model.y: batch_y,
                     model.x1: batch_x1, model.x2: batch_x2,
                     model.x_mask1: batch_mask1, model.x_mask2: batch_mask2}

        # loss, nb_right, _ = session.run(fetches, feed_dict)
        loss, prob, _ = session.run(fetches, feed_dict)
        loss_sum += loss * batch_size
        probs.append(prob)
        # nb_right_sum += nb_right

    # 如果训练集的样本数无法刚好被batch_size整除
    if nb_left > 0:
        feed_dict = {model.keep_prob: keep_prob, model.word_keep_prob: word_keep_prob, model.score_vec: score[-nb_left:][:],
                     # model.y: y[-nb_left:],
                     model.x1: x1[-nb_left:][:], model.x2: x2[-nb_left:][:],
                     model.x_mask1: get_mask(x1[-nb_left:][:]), model.x_mask2: get_mask(x2[-nb_left:][:])}

        # loss, nb_right, _ = session.run(fetches, feed_dict)
        loss, prob, _ = session.run(fetches, feed_dict)
        loss_sum += loss * nb_left
        probs.append(prob)
        # nb_right_sum += nb_right

    print("This epoch costs time {} s".format(time.time() - st_time))

    average_loss = loss_sum / nb_samples
    score = vector2score(score)
    probs = np.concatenate(probs, axis=0)
    pred_score = vector2score(probs)
    sp = my_spearman(pred_score, score)
    pe = my_pearson(pred_score, score)
    mse_ = mse(pred_score, score)
    # accuracy = nb_right_sum * 1.0 / nb_samples
    
    return average_loss, sp, pe, mse_
    # return average_loss, accuracy


def validate_or_test(session, model, data, batch_size):
    # 这里的data可能是验证集也可能是测试集，返回平均loss和正确率
    x1 = data["sent1"]
    x2 = data["sent2"]
    # y = data["label"]
    score = data["score"]

    nb_samples = x1.shape[0]
    nb_batches = int(math.floor(nb_samples * 1.0 / batch_size))
    nb_left = nb_samples - batch_size * nb_batches

    fetches = [model.loss, model.prob]
    # no training for validation/test set

    loss_sum = 0.0
    probs = []
    nb_right_sum = 0
    # print(nb_samples, nb_batches, nb_left)
    for j in range(nb_batches):
        batch_x1 = x1[j * batch_size: j * batch_size + batch_size][:]  # (batch_size, 78)
        batch_x2 = x2[j * batch_size: j * batch_size + batch_size][:]  # (batch_size, 78)
        batch_score = score[j * batch_size: j * batch_size + batch_size][:]  # (batch_size, 5)
        # batch_y = y[j * batch_size: j * batch_size + batch_size]  # (batch_size, )
        batch_mask1 = get_mask(batch_x1)
        batch_mask2 = get_mask(batch_x2)
        feed_dict = {model.keep_prob: 1.0, model.word_keep_prob: 1.0,
                     model.x1: batch_x1, model.x2: batch_x2, model.score_vec: batch_score,
                     # model.y: batch_y,
                     model.x_mask1: batch_mask1, model.x_mask2: batch_mask2
                     }

        # loss, nb_right = session.run(fetches, feed_dict)
        loss, prob = session.run(fetches, feed_dict)
        # nb_right_sum += nb_right
        loss_sum += loss * batch_size
        probs.append(prob)

    if nb_left > 0:
        feed_dict = {model.keep_prob: 1.0, model.word_keep_prob: 1.0,
                     model.x1: x1[-nb_left:][:], model.x2: x2[-nb_left:][:], model.score_vec: score[-nb_left:][:],
                     # model.y: y[-nb_left:],
                     model.x_mask1: get_mask(x1[-nb_left:][:]), model.x_mask2: get_mask(x2[-nb_left:][:]), 
                     }

        # loss, nb_right = session.run(fetches, feed_dict)
        loss, prob = session.run(fetches, feed_dict)
        # nb_right_sum += nb_right
        loss_sum += loss * nb_left
        probs.append(prob)

    average_loss = loss_sum / nb_samples
    probs = np.concatenate(probs, axis=0)
    pred_scores = vector2score(probs)
    score = vector2score(score)
    sp = my_spearman(pred_scores, score)
    pe = my_pearson(pred_scores, score)
    mse_ = mse(pred_scores, score)
    # accuracy = nb_right_sum * 1.0 / nb_samples

    return average_loss, sp, pe, mse_


def load_snli_data():
    data_path = "../SNLI/processed_snli.pkl"
    # 没有额外划分验证集，从训练集中随机抽出一小部分作为验证集,这里按 49000/4570 的样本数进行划分
    f = open(data_path, 'rb')
    data = pkl.load(f)
    f.close()

    train = data["train"]
    dev = data["dev"]
    test = data["test"]
    word2idx = data["voc"]

    return train, dev, test, word2idx


def load_sick_data():
    data_path = "../SICK/processed_sick.pkl"
    # 没有额外划分验证集，从训练集中随机抽出一小部分作为验证集,这里按 49000/4570 的样本数进行划分
    f = open(data_path, 'rb')
    data = pkl.load(f)
    f.close()

    train = data["train"]
    dev = data["dev"]
    test = data["test"]
    word2idx = data["voc"]

    return train, dev, test, word2idx


if __name__ == "__main__":
    train_data, dev_data, test_data, word2idx = load_sick_data()
    # get_word_emb(word2idx)
    # word_emb=get_pretrained_embedding("../SNLI/embedding_matrix.pkl", voc_size=57323)
    word_emb=get_pretrained_embedding("../SICK/embedding_matrix.pkl", voc_size=2461)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # model = models.SPM(word_emb=get_pretrained_embedding("embedding_matrix.pkl", voc_size=84598))
    sess = tf.Session()
    CONFIG = {"max_sentence_length1": 30,  
              "max_sentence_length2": 32,  # 如果有模型单独需要一些配置信息的话直接放上去就行了
              "emb_dim": 300,
              "voc_size": 2461,
              "preprocess": "lstm",
              "ave_mode": "window-sa",      # "sa", "window-sa", "gated-window-sa"
              "nb_classes": 5,
              "lr": 0.004,
              "word_emb": word_emb,
              "lstm_dim": 300
              }

    model = models.WSAN(config_=CONFIG)
    sess.run(tf.global_variables_initializer())

    nb_epoches = 100
    lr_decay = 0.8
    decay_start = 5

    # best_accuracy = [0.0, 0.0]
    best_valid_corr = []
    best_corr = [0.0, 0.0]
    for i in range(nb_epoches):
        if i >= decay_start:
            new_lr = sess.run(model.learning_rate) * lr_decay
            model.assign_lr(sess, new_lr)
            print("The new lr is {0}".format(new_lr))

        print("Epoch {0}".format(i))
        average_loss \
            = train_epoch(sess, model, batch_size=25, train_data=train_data, keep_prob=1.0, word_keep_prob=1.0)
        # print("ave loss: {0}, accuracy: {1}".format(average_loss, accuracy))
        print("ave loss: {0}".format(average_loss))

        devel_loss, devel_sp, devel_pe, devel_mse \
            = validate_or_test(sess, model, batch_size=300, data=dev_data)
        test_loss, test_sp, test_pe, test_mse \
            = validate_or_test(sess, model, batch_size=300, data=test_data)
        print("In devel set, ave loss: {0}, spearman: {1}, pearson: {2}, mse: {3}".format(devel_loss, devel_sp, devel_pe, devel_mse))
        print("In test set,  ave loss: {0}, spearman: {1}, pearson: {2}, mse: {3}".format(test_loss, test_sp, test_pe, test_mse))

        if (devel_sp + devel_pe)/2.0 > best_valid_corr:
            best_valid_corr = (devel_sp + devel_pe)/2.0
            best_corr = [test_sp, test_pe]
        elif (devel_sp + devel_pe)/2.0 == best_valid_corr:
            if test_sp + test_pe > best_corr[0] + best_corr[1]:
                best_corr = [test_sp, test_pe]
        else:
            pass


        # if devel_acc > best_accuracy[0]:
        #     best_accuracy = [devel_acc, test_acc]
        # elif devel_acc == best_accuracy[0]:
        #     best_accuracy[1] = max(test_acc, best_accuracy[1])
        # else:
        #     pass
        sys.stdout.flush()

    print("Test accuracy is {0}".format(best_accuracy[-1]))
    
