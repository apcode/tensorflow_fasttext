"""Predict classification on provided text.

Uses a SavedModel produced by classifier.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
from tensorflow.contrib.saved_model.python.saved_model import reader
from tensorflow.contrib.saved_model.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import loader

tf.flags.DEFINE_string("text", None, "Text to predict label of")
tf.flags.DEFINE_string("signature_def", None,
                       "Stored signature key of method to call (proba|embedding)")
tf.flags.DEFINE_string("saved_model", None, "Directory of SavedModel")
tf.flags.DEFINE_boolean("debug", False, "Debug")
FLAGS = tf.flags.FLAGS

_TAG = "serve"


def ProcessInput(text):
    tokens = word_tokenize(text)
    return np.array(tokens)


def RunModel(saved_model_dir, signature_def_key, text):
    saved_model = reader.read_saved_model(saved_model_dir)
    meta_graph =  None
    for meta_graph_def in saved_model.meta_graphs:
        if meta_graph_def.meta_info_def.tags == _TAG:
            meta_graph = meta_graph_def
    signature_def = signature_def_utils.get_signature_def_by_key(
        meta_graph, signature_def_key)
    inputs_feed_dict = {
        signature_def.inputs["inputs"].name: ProcessInput(text)
    }
    if signature_def_key == "proba":
        output_key = "scores"
    elif signature_def_key == "embedding":
        output_key = "outputs"
    else:
        raise ValueError("Unrecognised signature_def %s" % (signature_def_key))
    output_tensor = signature_def.outputs[output_key].name
    with tf.Session() as sess:
        loader.load(sess, [_TAG], saved_model_dir)
        outputs = sess.run(output_tensor,
                           feed_dict=inputs_feed_dict)
        return outputs


def main(_):
    if not FLAGS.text:
        raise ValueError("No --text provided")
    outputs = RunModel(FLAGS.saved_model, FLAGS.signature_def, FLAGS.text)
    if FLAGS.signature_def == "proba":
        print("Proba:", outputs[0])
        print("Class(1-N):", np.argmax(outputs[0]) + 1)
    elif FLAGS.signature_def == "embedding":
        print(outputs[0])


if __name__ == '__main__':
    if FLAGS.debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
    
