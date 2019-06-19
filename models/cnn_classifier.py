import tensorflow as tf

class Model:
    def __init__(self, hparams, dropout_keep_prob,word_embeddings, inputs: tf.Tensor, lengths: tf.Tensor):
        self.hparams = hparams
        self.dropout_keep_prob = dropout_keep_prob
        self.word_embeddings = word_embeddings

        self.logits = self._model_fn(inputs)

    def _model_fn(self, inputs: tf.Tensor):
        print("Building graph for model: CNN Classifier")

        # Number of possible output classes.
        output_dim = len(self.hparams.out_classes) # POS or NEG -> 2

        word_embeddings = tf.Variable(
            self.word_embeddings,
            name="word_embeddings",
            dtype=tf.float32,
            trainable=True
        )

        word_embedded = tf.nn.embedding_lookup(word_embeddings, inputs) # [batch_size, time, embed_dim]
        word_feature_map = tf.expand_dims(word_embedded, -1) # [batch_size, time, embed_dim, 1]

        # Convolution & Max pool
        features = []
        # filter_size : [3, 4, 5]
        for size in self.hparams.filter_size:
            with tf.variable_scope("CNN_filter_%d" % size):
                # Add padding to mark the beginning and end of words.
                pad_height = size - 1
                pad_shape = [[0, 0], [pad_height, pad_height], [0, 0], [0, 0]]
                word_feature_map = tf.pad(word_feature_map, pad_shape)

                feature = tf.layers.conv2d(
                    inputs=word_feature_map,
                    filters=self.hparams.num_filters,
                    kernel_size=[size, self.hparams.embedding_dim],
                    use_bias=True
                ) # [batch, time, 1, out_channels]

                feature = tf.reduce_max(feature, axis=1) # [batch, 1, out_channels]
                feature = tf.squeeze(feature, axis=1) # [batch, out_channels]
                features.append(feature)

        layer_out = tf.concat(features, axis=1) # [batch, out_channels * len(filter_size)]
        layer_out = tf.nn.dropout(layer_out, self.dropout_keep_prob)


        with tf.variable_scope("layer_out"):
            logits = tf.layers.dense(
                inputs=layer_out,
                units=output_dim,
                activation=None,
                kernel_initializer=tf.initializers.variance_scaling(
                    scale=2.0, mode="fan_in", distribution="normal")
            )

        return logits