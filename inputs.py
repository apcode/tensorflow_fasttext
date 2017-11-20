"""Input feature columns and input_fn for models.

Handles both training, evaluation and inference.
"""
import tensorflow as tf


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
        print("Input file:", input_file)
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
