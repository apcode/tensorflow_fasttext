"""Train simple fastText-style classifier.

Inputs:
  words - text to classify
  ngrams - n char ngrams for each word in words
  labels - output classes to classify

Model:
  word embedding
  ngram embedding
  LogisticRegression classifier of embeddings to labels
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inputs
import tensorflow as tf
from tensorflow.contrib.layers import feature_column
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators.run_config import RunConfig
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec


tf.flags.DEFINE_string("train_records", None,
                       "Training file pattern for TFRecords, can use wildcards")
tf.flags.DEFINE_string("eval_records", None,
                       "Evaluation file pattern for TFRecords, can use wildcards")
tf.flags.DEFINE_string("predict_records", None,
                       "File pattern for TFRecords to predict, can use wildcards")
tf.flags.DEFINE_integer("label_file", None, "File containing output labels")
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

tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for training")
tf.flags.DEFINE_float("clip_gradient_ratio", 5.0, "Clip gradient norm to this ratio")
tf.flags.DEFINE_integer("batch_size", 128, "Training minibatch size")
tf.flags.DEFINE_integer("train_steps", 1000,
                        "Number of train steps, None for continuous")
tf.flags.DEFINE_integer("eval_steps", 100, "Number of eval steps")
tf.flags.DEFINE_integer("checkpoint_steps", 1000,
                        "Steps between saving checkpoints")
tf.flags.DEFINE_integer("num_threads", 1, "Number of reader threads")
tf.flags.DEFINE_boolean("debug", False, "Turn on debug logging")
FLAGS = tf.flags.FLAGS


def FeatureColumns(mode):
    if not FLAGS.vocab_size:
        FLAGS.vocab_size = len(open(FLAGS.vocab_file).readlines())
    return inputs.FeatureColumns(
        mode, FLAGS.vocab_file, FLAGS.vocab_size, FLAGS.num_oov_vocab_buckets,
        FLAGS.embedding_dimension, FLAGS.num_ngram_buckets,
        FLAGS.ngram_embedding_dimension)


def Estimator(mode, feature_columns, model_dir, config=None):
    num_classes = len(open(FLAGS.label_file).readlines())
    model = tf.contrib.learn.LinearClassifier(
        feature_columns, model_dir, n_classes=num_classes,
        optimizer=tf.train.AdamOptimizer(learning_rate),
        gradient_clip_norm=FLAGS.clip_gradient,
        config=config)
    return model


def Experiment(model_dir):
    """Construct an experiment for training and evaluating a model.
    Saves checkpoints and exports the model for tf serving.
    """
    mode = tf.estimator.ModeKeys.TRAIN
    config = tf.estimators.RunConfig(
        save_checkpoints_secs=None,
        save_checkpoints_steps=FLAGS.checkpoint_steps)
    feature_columns = FeatureColumns(mode)
    train_input = inputs.InputFn(FLAGS.train_records, feature_columns,
                                 FLAGS.batch_size, mode, FLAGS.num_epochs)
    eval_input = inputs.InputFn(FLAGS.eval_records, feature_columns, FLAGS.batch_size,
                                mode, num_epochs=1)
    estimator = Estimator(mode, feature_columns, model_dir, config)
    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input,
        eval_input_fn=eval_input,
        train_steps=FLAGS.train_steps,
        eval_steps=FLAGS.eval_steps,
        eval_delay_secs=0,
        eval_metrics=None,
        continuous_eval_throttle_secs=10,
        min_eval_frequency=1000,
        train_monitors=None)
    return experiment


def Predict(model_dir):
    mode = tf.estimator.ModeKeys.PREDICT
    feature_columns = FeatureColumns(mode)
    predict_input = inputs.InputFn(FLAGS.predict_records, feature_columns,
                                   FLAGS.batch_size, mode, num_epochs=1)
    estimator = Estimator(mode, feature_columns, model_dir)
    results = estimator.predict(input_fn=predict_input)
    with tf.python_io.tf_record_iterator(FLAGS.predict_records) as records:
        for result, record in izip(results, records):
            print("%d (%d) %s" % (result, record["label"], record["words"]))


def main(_):
    if FLAGS.train_records:
        learn_runner.run(experiment_fn=Experiment, output_dir=FLAGS.model_dir)
    if FLAGS.predict_records:
        Predict(FLAGS.model_dir)


if __name__ == '__main__':
    if FLAGS.debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
