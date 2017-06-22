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
    features = set()
    word_ids = tf.feature_column.categorical_column_with_vocabulary_file(
        "words", vocab_file, vocab_size, num_oov_buckets=num_oov_vocab_buckets)
    words = tf.feature_column.embedding_column(
        word_ids, embedding_dimension, combiner='sum')
    features.add(words)
    if num_ngram_hash_buckets is not None:
        ngram_ids = tf.feature_column.categorical_column_with_hash_bucket(
            "ngrams", num_ngram_hash_buckets)
        ngrams = tf.feature_column.embedding_column(
            ngram_ids, ngram_embedding_dimension)
        features.add(ngrams)
    if mode != tf.estimator.ModeKeys.PREDICT:
        label = tf.feature_column.numeric_column("label", dtype=tf.int64)
        features.add(label)
    return features


def InputFn(input_file,
            feature_columns,
            batch_size,
            num_epochs=None,
            num_threads=1):
    feature_parse_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    def input_fn():
        features = tf.contrib.learn.read_batch_features(
            input_file, batch_size, feature_parse_spec,
            tf.TFRecordReader,
            num_epochs=1, reader_num_threads=num_threads)
        label = None
        if "label" in features:
            label = features.pop("label")
        return features, label
    return input_fn
