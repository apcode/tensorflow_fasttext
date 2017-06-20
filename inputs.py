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
        features["label"] = tf.feature_column.numeric_column("label")
    return features


def InputFn(input_file,
            feature_columns,
            batch_size,
            num_epochs=None,
            num_threads=1):
    def input_fn():
        features = tf.contrib.learn.read_batch_features(
            input_file, batch_size, feature_columns,
            tf.TFRecordReader,
            num_epochs=1, reader_num_threads=num_threads)
        label = None
        if "label" in features:
            label = features.pop("label")
        return features, label
    return input_fn
