# coding: utf-8
# author: huang ting
# created time: 2018-04-28-20:44
import numpy as np
import pickle as pkl
import os


label2idx = {"entailment": 0, "neutral": 1, "contradiction": 2, "-": 3}
label2idx_sick = {"ENTAILMENT": 0, "NEUTRAL": 1, "CONTRADICTION": 2, "-": 3}


def score2vector(score):
	# map the relatedness score to the 5-d vector
	vec = [0.0, 0.0, 0.0, 0.0, 0.0]
	if score == 5.0:
		vec = [0.0, 0.0, 0.0, 0.0, 1.0]
	else:
		floor = np.floor(score).astype(int)
		vec[floor] = score - floor
		vec[floor-1] = 1.0 - (score - floor)

	return np.array(vec, dtype=float)




def process_line(line):
	# 对一行进行处理，提取label, 得到句子1、句子2、label
	sample = line.strip().split('	')
	# print(len(sample))
	if sample[0] not in label2idx:
		print(line)
	label = label2idx[sample[0]]
	sentence1 = sample[5].lower().split()
	sentence2 = sample[6].lower().split()
	return sentence1, sentence2, label


def process_sick_line(line):
	# pair_ID	sentence_A	sentence_B	relatedness_score	entailment_judgment
	line =  line.strip().split('	')
	assert len(line) == 5
	sentence1 = line[1].lower().split()
	sentence2 = line[2].lower().split()
	score = float(line[3])
	score_vec = score2vector(score)
	label = label2idx_sick[line[-1]]

	return sentence1, sentence2, score_vec, label


def read_snli_file(file_path):
	sentences1 = []
	sentences2 = []
	labels = []
	f = open(file_path)
	count = 0
	for line in f:
		if count == 0:
			count += 1
			continue
		sent1, sent2, la = process_line(line)
		if la == 3:
			continue
		sentences1.append(sent1)
		sentences2.append(sent2)
		labels.append(la)

	f.close()

	return sentences1, sentences2, labels


def read_sick_file(file_path):
	sentences1 = []
	sentences2 = []
	labels = []
	scores = []
	f = open(file_path)
	count = 0
	for line in f:
		if count == 0:
			count += 1
			continue
		sent1, sent2, score_vec, la = process_sick_line(line)
		sentences1.append(sent1)
		sentences2.append(sent2)
		labels.append(la)
		scores.append(score_vec)

	f.close()

	return sentences1, sentences2, labels, scores


def map2idx_sequence(sentences, global_voc, max_length):
	nb_samples = len(sentences)
	array_x = np.zeros([nb_samples, max_length], dtype=int)
	for i in range(nb_samples):
		sent = sentences[i]
		idx_seq = []
		for word in sent:
			if word not in global_voc:
				global_voc[word] = len(global_voc)
			idx_seq.append(global_voc[word])
		array_x[i, :len(idx_seq)] = idx_seq
	return global_voc, array_x



def read_process_snli(file_paths=["../SNLI/snli_1.0_train.txt", "../SNLI/snli_1.0_dev.txt", "../SNLI/snli_1.0_test.txt"]):
	train_data = read_snli_file(file_paths[0])
	dev_data = read_snli_file(file_paths[1])
	test_data = read_snli_file(file_paths[-1])
	voc = {"#P#A#D#D#I#N#G#_1_2_3_4_5_lalala": 0}
	# train_sent1 = map2idx_sequence(train_data[0], voc, max_length)
	# train_sent1 = map2idx_sequence(train_data[0])
	sentsss1 = train_data[0] + dev_data[0] + test_data[0]
	sentsss2 = train_data[1] + dev_data[1] + test_data[1]
	l1 = np.array([len(it) for it in sentsss1], dtype=int)
	l2 = np.array([len(it) for it in sentsss2], dtype=int)

	print(l1.max(), l2.max())
	print(l1.mean(), l2.mean())
	print(len(train_data[0]), len(train_data[1]))
	print(len(dev_data[0]), len(dev_data[1]))
	print(len(test_data[0]), len(test_data[1]))

	voc, train_sent1 = map2idx_sequence(train_data[0], voc, 78)
	voc, train_sent2 = map2idx_sequence(train_data[1], voc, 56)
	voc, dev_sent1 = map2idx_sequence(dev_data[0], voc, 78)
	voc, dev_sent2 = map2idx_sequence(dev_data[1], voc, 56)
	voc, test_sent1 = map2idx_sequence(test_data[0], voc, 78)
	voc, test_sent2 = map2idx_sequence(test_data[1], voc, 56)

	print(len(voc))
	print(train_sent1.shape)
	print(train_sent2.shape)
	print(dev_sent1.shape)
	print(dev_sent2.shape)
	print(test_sent1.shape)
	print(test_sent2.shape)

	print(test_sent2[-5:])
	train_labels = np.array(train_data[-1], dtype=int)
	dev_labels = np.array(dev_data[-1], dtype=int)
	test_labels = np.array(test_data[-1], dtype=int)

	train = {"sent1": train_sent1, "sent2": train_sent2, "label": train_labels}
	dev = {"sent1": dev_sent1, "sent2": dev_sent2, "label": dev_labels}
	test = {"sent1": test_sent1, "sent2": test_sent2, "label": test_labels}

	processed_data = {"train": train, "dev": dev, "test": test, "voc": voc}
	f = open("../SNLI/processed_snli.pkl", "wb")
	pkl.dump(processed_data, f)
	f.close()


def read_process_sick(file_paths=["../SICK/SICK_train.txt", "../SICK/SICK_dev.txt", "../SICK/SICK_test.txt"]):
	train_data = read_sick_file(file_paths[0])
	dev_data = read_sick_file(file_paths[1])
	test_data = read_sick_file(file_paths[-1])
	voc = {"#P#A#D#D#I#N#G#_1_2_3_4_5_lalala": 0}

	sentsss1 = train_data[0] + dev_data[0] + test_data[0]
	sentsss2 = train_data[1] + dev_data[1] + test_data[1]
	l1 = np.array([len(it) for it in sentsss1], dtype=int)
	l2 = np.array([len(it) for it in sentsss2], dtype=int)

	print(l1.max(), l2.max())
	print(l1.mean(), l2.mean())
	print(len(train_data[0]), len(train_data[1]))
	print(len(dev_data[0]), len(dev_data[1]))
	print(len(test_data[0]), len(test_data[1]))

	voc, train_sent1 = map2idx_sequence(train_data[0], voc, 30)
	voc, train_sent2 = map2idx_sequence(train_data[1], voc, 32)
	voc, dev_sent1 = map2idx_sequence(dev_data[0], voc, 30)
	voc, dev_sent2 = map2idx_sequence(dev_data[1], voc, 32)
	voc, test_sent1 = map2idx_sequence(test_data[0], voc, 30)
	voc, test_sent2 = map2idx_sequence(test_data[1], voc, 32)

	train_labels = np.array(train_data[2], dtype=int)
	dev_labels = np.array(dev_data[2], dtype=int)
	test_labels = np.array(test_data[2], dtype=int)

	train_scores = np.array(train_data[-1], dtype=np.float)
	dev_scores = np.array(dev_data[-1], dtype=np.float)
	test_scores = np.array(test_data[-1], dtype=np.float)

	train = {"sent1": train_sent1, "sent2": train_sent2, "label": train_labels, "score": train_scores}
	dev = {"sent1": dev_sent1, "sent2": dev_sent2, "label": dev_labels, "score": dev_scores}
	test = {"sent1": test_sent1, "sent2": test_sent2, "label": test_labels, "score": test_scores}

	processed_data = {"train": train, "dev": dev, "test": test, "voc": voc}
	f = open("../SICK/processed_sick.pkl", "wb")
	pkl.dump(processed_data, f)
	f.close()

if __name__ == "__main__":
	# read_process_snli()
	read_process_sick()
	