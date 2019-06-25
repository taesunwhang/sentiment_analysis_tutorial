import os
import logging
import math
from datetime import datetime
import tensorflow as tf

from data_process import *

class TextClassifier:
    def __init__(self, hparams):
        self.hparams = hparams
        self.data_dir = hparams.data_dir
        #logger
        self._logger = logging.getLogger(__name__)
        self._make_data_processor()

    def _make_data_processor(self):
        processors = {
            "sentiment_analysis": AmazonReviewsProcessor,
        }

        data_dir = self.hparams.data_dir
        self.processor = processors[self.hparams.task_name](self.hparams)
        self.train_examples = self.processor.get_train_examples(data_dir)
        self.test_examples = self.processor.get_test_examples(data_dir)
        self.label_list = self.processor.get_labels()

        self.word_embeddings = self.processor.get_word_embeddings()

    def _make_placeholder(self):
        # [batch_size, max_seq_len]
        self.inputs_ph = tf.placeholder(tf.int32, shape=[None, None], name="inputs_ph")
        self.lengths_ph = tf.placeholder(tf.int32, shape=[None], name="lengths_ph")
        self.labels_ph = tf.placeholder(tf.int32, shape=[None], name="labels_ph")

        self._dropout_keep_prob_ph = tf.placeholder(tf.float32, shape=[], name="dropout_keep_prob")

    def _make_feed_dict(self, batch_data, dropout_keep_prob):
        feed_dict = {}
        batch_inputs, batch_lengths, batch_labels = batch_data

        # word-level
        feed_dict[self.inputs_ph] = batch_inputs
        feed_dict[self.labels_ph] = batch_labels
        feed_dict[self.lengths_ph] = batch_lengths

        feed_dict[self._dropout_keep_prob_ph] = dropout_keep_prob

        return feed_dict

    def _build_graph(self):

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # model_fn -> cnn_classifier
        with tf.variable_scope("mode_fn", reuse=False):
            model = self.hparams.graph.Model(self.hparams, self._dropout_keep_prob_ph,
                                             self.word_embeddings, self.inputs_ph, self.lengths_ph)
            self.logits = model.logits

        with tf.name_scope("cross_entropy"):
            loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels_ph,
                                                                     name="cross_entropy")
            self.loss_op = tf.reduce_mean(loss_op, name='cross_entropy_mean')
            self.train_op = tf.train.AdamOptimizer().minimize(loss_op, global_step=self.global_step)

        eval = tf.nn.in_top_k(self.logits, self.labels_ph, 1)
        correct_count = tf.reduce_sum(tf.cast(eval, tf.int32))
        with tf.name_scope("accuracy"):
            self.accuracy = tf.divide(correct_count, tf.shape(self.labels_ph)[0])

    def train(self):
        self.sess = tf.Session()

        with self.sess.as_default():
            # build placeholder
            self._make_placeholder()
            # build train graph
            self._build_graph()

            # checkpoint file saver
            saver = tf.train.Saver()
            total_data_len = int(math.ceil(len(self.train_examples) / self.hparams.train_batch_size))
            self._logger.info("Batch iteration per epoch is %d" % total_data_len)

            tf.global_variables_initializer().run()
            start_time = datetime.now().strftime('%H:%M:%S')
            self._logger.info("Start train model at %s" % start_time)
            for epoch_completed in range(self.hparams.num_epochs):

                if epoch_completed > 0 and self.hparams.training_shuffle_num > 0:
                    self.train_examples = self.processor.get_train_examples(self.hparams.data_dir)

                step_loss_mean, step_accuracy_mean, print_step_count = 0, 0, 0
                for i in range(total_data_len):
                    batch_data = self.processor.get_batch_data(i, self.hparams.train_batch_size, "train")

                    accuracy_val, loss_val, global_step_val, _ = self.sess.run(
                        [self.accuracy,
                         self.loss_op,
                         self.global_step,
                         self.train_op],
                        feed_dict=self._make_feed_dict(batch_data, self.hparams.dropout_keep_prob)
                    )
                    step_loss_mean += loss_val
                    step_accuracy_mean += accuracy_val
                    print_step_count += 1

                    if global_step_val % self.hparams.print_step == 0:
                        step_loss_mean /= print_step_count
                        step_accuracy_mean /= print_step_count
                        self._logger.info("[Step %d] loss: %.4f, accuracy: %.2f%%" % (
                            global_step_val, step_loss_mean, step_accuracy_mean * 100))

                        step_loss_mean, step_accuracy_mean, print_step_count = 0, 0, 0

                self._logger.info("End of epoch %d." % (epoch_completed + 1))
                save_path = saver.save(self.sess, "%s/model_ckpt/model.ckpt" % self.hparams.root_dir, global_step=global_step_val)
                self._run_evaluate()


                self._logger.info("Model saved at: %s" % save_path)

    def _run_evaluate(self):
        self._logger.info("---Evaluation Phase---")
        total_data_len = int(math.ceil(len(self.test_examples) / self.hparams.eval_batch_size))
        self._logger.info("Evaluation batch iteration per epoch is %d" % total_data_len)

        tot_correct_count = 0
        for i in range(total_data_len):
            batch_data = self.processor.get_batch_data(i, self.hparams.eval_batch_size, "test")

            logits_val, loss_val, global_step_val = self.sess.run(
                [self.logits, self.loss_op, self.global_step],
                feed_dict=self._make_feed_dict(batch_data, 1.0)
            )
            correct_count = np.sum(np.equal(np.argmax(logits_val, axis=-1), np.array(batch_data[2])))
            tot_correct_count += correct_count

        self._logger.info("Test Accuracy : %.2f%%" % ((tot_correct_count/len(self.test_examples))*100))

