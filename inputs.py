"""Input feature columns and input_fn for models.

Handles both training, evaluation and inference.
"""
import tensorflow as tf


def FeatureColumns(mode,
                   vocab_file,
                   vocab_size,
                   num_oov_vocab_buckets,
                   embedding_dimension,
                   num_ngram_hash_buckets=None,
                   ngram_embedding_dimension=None):
    word_ids = tf.feature_column.categorical_column_with_vocabulary_file(
        "words", vocab_file, vocab_size, num_oov_buckets=num_oov_vocab_buckets)
    words = tf.feature_column.embedding_column(
        word_ids, embedding_dimension, combiner='sum')
    ngrams = None
    if num_ngram_hash_buckets is not None:
        ngram_ids = tf.feature_column.categorical_column_with_hash_bucket(
            "ngrams", num_ngram_hash_buckets)
        ngrams = tf.feature_column.embedding_column(
            ngram_ids, ngram_embedding_dimension)
    features = {"words": words}
    if ngrams:
        features["ngrams"] = ngrams
    if mode != tf.estimator.ModeKeys.PREDICT:
        features["labels"] = tf.feature_column.numerical_column("labels")
    return features


def InputFn(input_file,
            features,
            batch_size,
            num_epochs=None,
            num_threads=1):
    def input_fn():
        features = tf.contrib.learn.read_batch_features(
            input_file, batch_size, features,
            tf.python_io.TFRecordReader,
            num_epochs=1, num_reader_threads=num_threads)
        labels = None
        if "labels" in features:
            labels = features.pop("label")
        return features, labels
    return input_fn
