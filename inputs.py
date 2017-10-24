"""Input feature columns and input_fn for models.

Handles both training, evaluation and inference.
"""
import tensorflow as tf


def FeatureColumns(include_target,
                   use_ngrams,
                   vocab_file,
                   vocab_size,
                   embedding_dimension,
                   num_oov_vocab_buckets,
                   label_file,
                   label_size,
                   ngram_embedding_dimension=None,
                   num_ngram_hash_buckets=None):
    features = []
    word_ids = tf.feature_column.categorical_column_with_vocabulary_file(
        "text", vocab_file, vocab_size, num_oov_buckets=num_oov_vocab_buckets)
    words = tf.feature_column.embedding_column(
        word_ids, embedding_dimension, combiner='sum')
    features.append(words)
    if use_ngrams:
        ngram_ids = tf.feature_column.categorical_column_with_hash_bucket(
            "ngrams", num_ngram_hash_buckets)
        ngrams = tf.feature_column.embedding_column(
            ngram_ids, ngram_embedding_dimension)
        features.append(ngrams)
    if include_target:
        label = tf.feature_column.categorical_column_with_vocabulary_file(
            "label", label_file, label_size)
        features.append(label)
    return set(features)


def InputFn(mode,
            use_ngrams,
            input_file,
            vocab_file,
            vocab_size,
            embedding_dimension,
            num_oov_vocab_buckets,
            label_file,
            label_size,
            ngram_embedding_dimension,            
            num_ngram_hash_buckets,
            batch_size,
            num_epochs=None,
            num_threads=1):
    if num_epochs <= 0:
        num_epochs=None
    def input_fn():
        include_target =  mode != tf.estimator.ModeKeys.PREDICT
        parse_spec = {"text": tf.VarLenFeature(dtype=tf.string)}
        if use_ngrams:
            parse_spec["ngrams"] = tf.VarLenFeature(dtype=tf.string)
        if include_target:
            parse_spec["label"] = tf.FixedLenFeature(shape=(1,), dtype=tf.string,
                                                     default_value=None)
        print("ParseSpec", parse_spec)
        features = tf.contrib.learn.read_batch_features(
            input_file, batch_size, parse_spec, tf.TFRecordReader,
            num_epochs=num_epochs, reader_num_threads=num_threads)
        features["text"] = tf.sparse_tensor_to_dense(features["text"],
                                                     default_value=" ")
        if use_ngrams:
            features["ngrams"] = tf.sparse_tensor_to_dense(features["ngrams"],
                                                           default_value=" ")
        label = None
        if include_target:
            label = features.pop("label")
        return features, label
    return input_fn
