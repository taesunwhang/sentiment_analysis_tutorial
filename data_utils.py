import os
import numpy as np
import random
import nltk

nltk.download('punkt')

data_dir = "./amazon_reviews"
glove_dir = "./glove.6B"
embedding_dim = 300

def make_data_samples():

	# 3,600,000 : train, 400,000 : test -> 90,000 : train, 10,000 : test
	data_type = {"train" : 18000, "test" : 2000}

	for t in data_type.keys():
		vocab = set()
		with open(os.path.join(data_dir, "%s.ft.txt" % t), "r", encoding='utf-8') as fr_handle:
			sentences = [line.strip() for line in fr_handle if len(line.strip()) > 1]
			for i in range(10):
				print(i + 1, "th shuffling has finished!")
				random.shuffle(sentences)

			sampled_data = random.sample(sentences, data_type[t])
			label_1_count = 0
			label_2_count = 0
			for data in sampled_data:
				label = data.split(' ')[0]
				if label == "__label__1":
					label_1_count += 1
				elif label == "__label__2":
					label_2_count += 1

			print(t, ":", len(sampled_data), "label 1 : %d, label 2 : %d" % (label_1_count, label_2_count))

		with open(os.path.join(data_dir, "%s.txt" % t), "w", encoding="utf-8") as fw_handle:
			for line in sampled_data:
				fw_handle.write(line+'\n')

def make_vocab_file():
	data_type = ["test", "train"]
	total_vocab = set()
	index = 0
	for t in data_type:
		vocab = set()
		with open(os.path.join(data_dir, "%s.txt" % t), "r", encoding='utf-8') as fr_handle:
			sentences = [line.strip() for line in fr_handle if len(line.strip()) > 1]
			print(t, ":", len(sentences))

			for sent in sentences:
				assert sent.split(' ')[0] not in [0, 1]

				sentence = ' '.join(sent.split(' ')[1:])
				vocab.update(nltk.word_tokenize(sentence))
				index += 1
				if index % 10000 == 0:
					print(index)

		print("# %s vocab : " % t, len(vocab))
		total_vocab = total_vocab.union(vocab)

	print("# total vocab : ", len(total_vocab))
	with open(os.path.join(data_dir, "vocab.txt"), "w", encoding="utf-8") as fw_handle:
		total_vocab = list(total_vocab)
		for word in total_vocab:
			word = word.lower()  # make a word lowercase
			fw_handle.write(word + '\n')

def export_trimmed_glove_vectors():
	glove_embedding = dict()
	glove_vocab = set()

	print("Making glove trimmed.npz, It will take few minutes...")
	with open(os.path.join(glove_dir, "glove.6B.300d.txt"), "r", encoding='utf-8') as f_handle:
		for line_idx, line in enumerate(f_handle):
			line = line.strip().split(" ")
			glove_word = line[0]
			glove_vocab.add(glove_word)
			glove_embedding[glove_word] = [float(x) for x in line[1:]]

		print("# glove vocab : ", len(glove_embedding))

	with open(os.path.join(data_dir, "vocab.txt"), "r") as f_handle:
		total_vocab = [line.strip() for line in list(f_handle)]
		print("# total vocab : ", len(total_vocab))

	# print("%d words are in Glove..." % len(glove_vocab.intersection(set(total_vocab))))

	embeddings = np.zeros([len(total_vocab) + 1, embedding_dim])

	for word_idx, word in enumerate(total_vocab):
		try:
			embeddings[word_idx] = np.asarray(glove_embedding[word.lower()])
		except KeyError:
			embeddings[word_idx] = np.random.uniform(-1, 1, embedding_dim)

	np.savez_compressed(os.path.join(glove_dir, "glove.6B.300d.trimmed.npz"), embeddings=embeddings)
	print("Save Glove Embeddings as an Numpy Array : ", len(embeddings))

if __name__ == '__main__':
		make_data_samples()
		make_vocab_file()
		export_trimmed_glove_vectors()