import os
import random
import numpy as np

import nltk
class InputExample(object):
	"""A single training/test example for simple sequence classification."""

	def __init__(self, input_text, label):
		self.input_text = input_text
		self.label = label

class AmazonReviewsProcessor(object):
	"""
	This code is implemented based on:
	https://github.com/google-research/bert/blob/master/run_classifier.py
	"""
	def __init__(self, hparams):
		self.hparams = hparams
		self._get_word_dict()

	def get_train_examples(self, data_dir):
		train_data = self._read_data(os.path.join(data_dir, "%s.txt" % "train"), shuffle=True)
		train_example = self._create_examples(train_data, "train")
		self.train_example = train_example

		return train_example

	def get_test_examples(self, data_dir):
		test_data = self._read_data(os.path.join(data_dir, "%s.txt" % "test"), shuffle=False)
		test_example = self._create_examples(test_data, "test")
		self.test_example = test_example

		return test_example

	def get_labels(self):
		"""See base class."""
		self.label_list = ["__label__1", "__label__2"]
		return self.label_list

	def _data_shuffling(self, inputs, shuffle_num):
		random_seed = random.sample(list(range(0, 1000)), 1)[0]
		print("Random Seed : ", random_seed)
		random.seed(random_seed)
		for i in range(shuffle_num):
			# print(i + 1, "th shuffling has finished!")
			random.shuffle(inputs)
		print("Shuffling total %d process is done! Total dialog context : %d" % (shuffle_num, len(inputs)))

		return inputs

	def _read_data(self, data_dir, shuffle=False):
		print("[Reading %s]" % data_dir)
		with open(data_dir, "r", encoding="utf-8") as fr_handle:
			total_data = [line.strip() for line in fr_handle if len(line.strip()) > 1]

		if shuffle and self.hparams.training_shuffle_num > 1:
			total_data = self._data_shuffling(total_data, self.hparams.training_shuffle_num)

		return total_data

	def _create_examples(self, inputs, set_type):
		"""Creates examples for the training and dev sets."""
		examples = []
		for (i, input_data) in enumerate(inputs):
			input_text = ' '.join(input_data.split(' ')[1:])
			label = str(input_data.split(' ')[0])

			examples.append(
				InputExample(input_text=input_text, label=label))
		print("%s data creation is finished! %d" % (set_type, len(examples)))

		return examples

	def _get_word_dict(self):
		with open(self.hparams.vocab_path) as f_handle:
			self.vocab = f_handle.read().splitlines()

		self.vocab.insert(0, "[PAD]")
		self.vocab.append("[UNK]")

		self.word2id = dict()
		for idx, word in enumerate(self.vocab):
			self.word2id[word] = idx

	def get_word_embeddings(self):
		with np.load(self.hparams.glove_embedding_path) as data:
			print("glove embedding shape", np.shape(data["embeddings"]))
			return data["embeddings"]

	def get_batch_data(self, curr_index, batch_size, set_type="train"):
		inputs = []
		lengths = []
		label_ids = []

		examples = {
			"train": self.train_example,
			"test": self.test_example
		}
		example = examples[set_type]

		for index, each_example in enumerate(example[curr_index * batch_size:batch_size * (curr_index + 1)]):
			tokenized_inputs_id, input_length, label_id = \
				convert_single_example(each_example, self.label_list, word2id=self.word2id)

			inputs.append(tokenized_inputs_id)
			lengths.append(input_length)
			label_ids.append(label_id)

		pad_inputs = rank_2_pad_process(inputs)

		return pad_inputs, lengths, label_ids


def rank_2_pad_process(inputs, special_id=False):
	append_id = 0
	if special_id:
		append_id = -1 # user_id -> -1

	max_sent_len = 0
	for sent in inputs:
		max_sent_len = max(len(sent), max_sent_len)

	# print("text_a max_lengths in a batch", max_sent_len)
	padded_result = []
	sent_buffer = []
	for sent in inputs:
		for i in range(max_sent_len - len(sent)):
			sent_buffer.append(append_id)
		sent.extend(sent_buffer)
		padded_result.append(sent)
		sent_buffer = []

	return padded_result

def convert_single_example(example, label_list, word2id):
	tokenized_input = nltk.word_tokenize(example.input_text)

	if len(tokenized_input) > 320:
		tokenized_input = tokenized_input[0:320]

	for word_idx, word_token in enumerate(tokenized_input):
		tokenized_input[word_idx] = word_token.lower()

	for a_idx, token_a in enumerate(tokenized_input):
		tokenized_input[a_idx] = word2id[token_a]
	input_length = len(tokenized_input)
	tokenized_input_id = tokenized_input

	label_map = {}
	for (i, label) in enumerate(label_list):
		label_map[label] = i
	label_id = label_map[example.label]

	return tokenized_input_id, input_length, label_id