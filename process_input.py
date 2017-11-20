"""Process input data into tensorflow examples, to ease training.

Input data is in one of two formats:
- facebook's format used in their fastText library.
- two text files, one with input text per line, the other a label per line.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import tensorflow as tf
import inputs
import text_utils
from collections import Counter
from six.moves import zip


tf.flags.DEFINE_string("facebook_input", None,
                       "Input file in facebook train|test format")
tf.flags.DEFINE_string("text_input", None,
                       """Input text file containing one text phrase per line.
                       Must have --labels defined
                       Used instead of --facebook_input""")
tf.flags.DEFINE_string("labels", None,
                       """Input text file containing one label for
                       classification  per line.
                       Must have --text_input defined.
                       Used instead of --facebook_input""")
tf.flags.DEFINE_string("ngrams", None,
                       "list of ngram sizes to create, e.g. --ngrams=2,3,4,5")
tf.flags.DEFINE_string("output_dir", ".",
                       "Directory to store resulting vector models and checkpoints in")
tf.flags.DEFINE_integer("num_shards", 1,
                        "Number of outputfiles to create")
FLAGS = tf.flags.FLAGS


def ParseFacebookInput(inputfile, ngrams):
    """Parse input in the format used by facebook FastText.
    labels are formatted as __label__1
    where the label values start at 0.
    """
    examples = []
    for line in open(inputfile):
        words = line.split()
        # label is first field with __label__ removed
        match = re.match(r'__label__(.+)', words[0])
        label = match.group(1) if match else None
        # Strip out label and first ,
        first = 2 if words[1] == "," else 1
        words = words[first:]
        examples.append({
            "text": words,
            "label": label
        })
        if ngrams:
            examples[-1]["ngrams"] = text_utils.GenerateNgrams(words, ngrams)
    return examples


def ParseTextInput(textfile, labelsfie, ngrams):
    """Parse input from two text files: text and labels.
    labels are specified 0-offset one per line.
    """
    examples = []
    with open(textfile) as f1, open(labelsfile) as f2:
        for text, label in zip(f1, f2):
            words = text_utils.TokenizeText(text)
            examples.append({
                "text": words,
                "label": label,
            })
            if ngrams:
                examples[-1]["ngrams"] = text_utils.GenerateNgrams(words, ngrams)
    return examples


def WriteExamples(examples, outputfile, num_shards):
    """Write examles in TFRecord format.
    Args:
      examples: list of feature dicts.
                {'text': [words], 'label': [labels]}
      outputfile: full pathname of output file
    """
    shard = 0
    num_per_shard = len(examples) / num_shards + 1
    for n, example in enumerate(examples):
        if n % num_per_shard == 0:
            shard += 1
            writer = tf.python_io.TFRecordWriter(outputfile + '-%d-of-%d' % \
                                                 (shard, num_shards))
        record = inputs.BuildTextExample(
            example["text"], example.get("ngrams", None), example["label"])
        writer.write(record)


def WriteVocab(examples, vocabfile, labelfile):
    words = Counter()
    labels = set()
    for example in examples:
        words.update(example["text"])
        labels.add(example["label"])
    with open(vocabfile, "w") as f:
        # Write out vocab in most common first order
        # We need this as NCE loss in TF uses Zipf distribution
        for word in words.most_common():
            f.write(word[0] + '\n')
    with open(labelfile, "w") as f:
        labels = sorted(list(labels))
        for label in labels:
            f.write(str(label) + '\n')


def main(_):
    # Check flags
    if not (FLAGS.facebook_input or (FLAGS.text_input and FLAGS.labels)):
        print >>sys.stderr, \
            "Error: You must define either facebook_input or both text_input and labels"
        sys.exit(1)
    ngrams = None
    if FLAGS.ngrams:
        ngrams = text_utils.ParseNgramsOpts(FLAGS.ngrams)
    if FLAGS.facebook_input:
        inputfile = FLAGS.facebook_input
        examples = ParseFacebookInput(FLAGS.facebook_input, ngrams)
    else:
        inputfile = FLAGS.text_input
        examples = ParseTextInput(FLAGS.text_input, FLAGS.labels, ngrams)
    outputfile = os.path.join(FLAGS.output_dir, inputfile + ".tfrecords")
    WriteExamples(examples, outputfile, FLAGS.num_shards)
    vocabfile = os.path.join(FLAGS.output_dir, inputfile + ".vocab")
    labelfile = os.path.join(FLAGS.output_dir, inputfile + ".labels")
    WriteVocab(examples, vocabfile, labelfile)


if __name__ == '__main__':
    tf.app.run()
