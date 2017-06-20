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

from inputs import FeatureColumns, InputFn
from model import Estimator

tf.flags.DEFINE_string("input", None,
                       "TFRecord file of text to classify")
tf.flags.DEFINE_integer("num_classes", None, "Number of output classes")
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


def ClassifyTFRecords(inputfile, model_dir):
    estimator = Estimator(
        tf.estimators.ModeKeys.PREDICT,
        FLAGS.vocab_size,
        FLAGS.vocab_file,
        FLAGS.num_oov_vocab_buckets,
        FLAGS.embedding_dimension,
        FLAGS.num_ngram_buckets,
        FLAGS.ngram_embedding_dimension,
        FLAGS.batch_size,
        num_epochs=1,
        model_dir,
        FLAGS.learning_rate,
        FLAGS.clip_gradient,
        FLAGS.num_classes,
        eval_records=FLAGS.input)
    results = estimator.predict(input_fn=eval_input, as_iterable=True)
    with tf.python_io.tf_record_iterator(inputfile) as records:
        for result, record in izip(results, records):
            print("%d (%d) %s" % (result, record["label"], record["words"]))


def main(_):
    ClassifyTFRecords(FLAGS.input, FLAGS.model_dir)


if __name__ == '__main__':
    if FLAGS.debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
