from collections import defaultdict
from models import cnn_classifier

BASE_PARAMS = defaultdict(
		model = "cnn_classifier",
		graph = cnn_classifier,
		root_dir = "./runs/cnn_classifier",
		out_classes=["POS","NEG"],
		num_epochs = 10,
		train_batch_size = 64,
		eval_batch_size = 200,
		training_shuffle_num = 50,
		filter_size = [3,4,5],
		num_filters = 100,
		glove_dir = "glove.6B",
		embedding_dim = 300,
		dropout_keep_prob = 0.8,

		glove_embedding_path = "./glove.6B/glove.6B.300d.trimmed.npz",
		vocab_path = "./amazon_reviews/vocab.txt",
		task_name="sentiment_analysis"
)

CNN_CLASSIFIER = BASE_PARAMS.copy()
CNN_CLASSIFIER.update(
		model="cnn_classfier",
		root_dir = "./runs/cnn_classifier",
		data_dir = "./amazon_reviews/",
		graph = cnn_classifier
)
