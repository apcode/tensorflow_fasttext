"""Predict classifications for text using a simple fastText-style classifier.

Inputs from tfrecords:
  words - text to classify
  ngrams - n char ngrams for each word in words
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from itertools import izip
from tensorflow.contrib.layers import feature_column

tf.flags.DEFINE_string("input", None,
                       "TFRecord file of text to classify")
tf.flags.DEFINE_string("vocab_file", None, "Vocabulary file, one word per line")
tf.flags.DEFINE_integer("vocab_size", None, "Number of words in vocabulary")
tf.flags.DEFINE_integer("num_oov_vocab_buckets", 20,
                        "Number of hash buckets to use for OOV words")
tf.flags.DEFINE_string("model_dir", ".",
                       "Output directory for checkpoints and summaries")

tf.flags.DEFINE_integer("embedding_dimension", 10, "Dimension of word embedding")
tf.flags.DEFINE_integer("num_ngram_buckets", 1000000,
                        "Number of hash buckets for ngrams")
tf.flags.DEFINE_integer("ngram_embedding_dimension", 10, "Dimension of word embedding")

tf.flags.DEFINE_integer("batch_size", 128, "Training minibatch size")
tf.flags.DEFINE_integer("num_threads", 1, "Number of reader threads")
tf.flags.DEFINE_boolean("debug", False, "Turn on debug logging")
FLAGS = tf.flags.FLAGS


def FeatureColumns(vocab_file,
                   vocab_size,
                   num_oov_vocab_buckets,
                   embedding_dimension,
                   num_ngram_hash_buckets=None,
                   ngram_embedding_dimension=None):
    word_ids = feature_column.categorical_column_with_vocabulary_file(
        "words", vocab_file, vocab_size, num_oov_buckets=num_oov_vocab_buckets)
    words = feature_column.embedding_column(
        word_ids, embedding_dimension, combiner='sum')
    ngrams = None
    if num_ngram_hash_buckets is not None:
        ngram_ids = feature_column.categorical_column_with_hash_bucket(
            "ngrams", num_ngram_hash_buckets)
        ngrams = feature_column.embedding_column(
            ngram_ids, ngram_embedding_dimension)
    features = {"words": words}
    if ngrams:
        features["ngrams"] = ngrams
    return features


def InputFn(input_file, features, num_epochs=None):
    def input_fn():
        features = tf.contrib.learn.read_batch_features(
            input_file, FLAGS.batch_size, features,
            tf.python_io.TFRecordReader,
            num_epochs=1, num_reader_threads=FLAGS.num_threads)
        return features, None
    return input_fn


def Estimator(model_dir):
    """Construct an Estimator for a classification model.
    Loads model from checkpoints.
    """
    if FLAGS.vocab_size is None:
        FLAGS.vocab_size = len(open(FLAGS.vocab_file).readlines())
    feature_columns = FeatureColumns(
        FLAGS.vocab_file, FLAGS.vocab_size, FLAGS.num_oov_vocab_buckets,
        FLAGS.embedding_dimension, FLAGS.num_ngram_buckets,
        FLAGS.ngram_embedding_dimension)
    eval_input = InputFn(FLAGS.eval_input, feature_columns)
    num_classes = len(open(FLAGS.label_file).readlines())
    model = tf.contrib.learn.LinearClassifier(
        feature_columns, model_dir, n_classes=FLAGS.num_classes,
        optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate),
        gradient_clip_norm=FLAGS.clip_gradient,
        config=config)
    return model
    

def ClassifyTFRecords(inputfile, model_dir):
    estimator = Estimator(model_dir)
    results = estimator.predict(input_fn=eval_input, as_iterable=True)
    with tf.python_io.tf_record_iterator(inputfile) as records:
        for result, record in izip(results, records):
            print("%d (%d) %s" % (result, record["label"], record["words"]))


def main(_):
    if FLAGS.input:
        ClassifyTFRecords(FLAGS.input, FLAGS.model_dir)


if __name__ == '__main__':
    if FLAGS.debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
