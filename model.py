"""Build Estimator for classifier.
"""
import tensorflow as tf
from inputs import FeatureColumns, InputFn


def Estimator(mode,
              vocab_size,
              vocab_file,
              num_oov_vocab_buckets,
              embedding_dimension,
              num_ngram_buckets,
              ngram_embedding_dimension,
              batch_size,
              num_epochs,
              model_dir,
              learning_rate,
              clip_gradient,
              num_classes=None,
              train_records=None,
              eval_records=None,
              config=None,
              label_file=None):
    """Construct an Estimator for a classification model.
    Loads model from checkpoints.
    """
    if vocab_size is None:
        vocab_size = len(open(vocab_file).readlines())
    feature_columns = FeatureColumns(
        mode, vocab_file, vocab_size, num_oov_vocab_buckets,
        embedding_dimension, num_ngram_buckets,
        ngram_embedding_dimension)
    if mode == tf.estimators.ModeKeys.TRAIN:
        train_input = InputFn(train_records, feature_columns, batch_size,
                              mode, num_epochs)
    eval_input = InputFn(eval_records, feature_columns, batch_size,
                         mode, num_epochs=1)
    if not num_classes:
        num_classes = len(open(label_file).readlines())
    model = tf.contrib.learn.LinearClassifier(
        feature_columns, model_dir, n_classes=num_classes,
        optimizer=tf.train.AdamOptimizer(learning_rate),
        gradient_clip_norm=clip_gradient,
        config=config)
    return model
