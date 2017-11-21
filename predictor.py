"""Predict classification on provided text.

Uses a SavedModel produced by classifier.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import inputs
import text_utils
from tensorflow.contrib.saved_model.python.saved_model import reader
from tensorflow.contrib.saved_model.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import loader

tf.flags.DEFINE_string("text", None, "Text to predict label of")
tf.flags.DEFINE_string("ngrams", None, "List of ngram lengths, E.g. --ngrams=2,3,4")
tf.flags.DEFINE_string("signature_def", "proba",
                       "Stored signature key of method to call (proba|embedding)")
tf.flags.DEFINE_string("saved_model", None, "Directory of SavedModel")
tf.flags.DEFINE_string("tag", "serve", "SavedModel tag, serve|gpu")
tf.flags.DEFINE_boolean("debug", False, "Debug")
FLAGS = tf.flags.FLAGS


def RunModel(saved_model_dir, signature_def_key, tag, text, ngrams_list=None):
    saved_model = reader.read_saved_model(saved_model_dir)
    meta_graph =  None
    for meta_graph_def in saved_model.meta_graphs:
        if tag in meta_graph_def.meta_info_def.tags:
            meta_graph = meta_graph_def
            break
    if meta_graph_def is None:
        raise ValueError("Cannot find saved_model with tag" + tag)
    signature_def = signature_def_utils.get_signature_def_by_key(
        meta_graph, signature_def_key)
    text = text_utils.TokenizeText(text)
    ngrams = None
    if ngrams_list is not None:
        ngrams_list = text_utils.ParseNgramsOpts(ngrams_list)
        ngrams = text_utils.GenerateNgrams(text, ngrams_list)
    example = inputs.BuildTextExample(text, ngrams=ngrams)
    example = example.SerializeToString()
    inputs_feed_dict = {
        signature_def.inputs["inputs"].name: [example],
    }
    if signature_def_key == "proba":
        output_key = "scores"
    elif signature_def_key == "embedding":
        output_key = "outputs"
    else:
        raise ValueError("Unrecognised signature_def %s" % (signature_def_key))
    output_tensor = signature_def.outputs[output_key].name
    with tf.Session() as sess:
        loader.load(sess, [tag], saved_model_dir)
        outputs = sess.run(output_tensor,
                           feed_dict=inputs_feed_dict)
        return outputs


def main(_):
    if not FLAGS.text:
        raise ValueError("No --text provided")
    outputs = RunModel(FLAGS.saved_model, FLAGS.signature_def, FLAGS.tag,
                       FLAGS.text, FLAGS.ngrams)
    if FLAGS.signature_def == "proba":
        print("Proba:", outputs)
        print("Class(1-N):", np.argmax(outputs) + 1)
    elif FLAGS.signature_def == "embedding":
        print(outputs[0])


if __name__ == '__main__':
    if FLAGS.debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
    
