import sys
import tensorflow as tf
from google.protobuf import text_format
from inputs import FeatureColumns, InputFn

VOCAB_FILE='/home/alan/Workspace/other/fastText/data/ag_news.train.vocab'
VOCAB_SIZE=95810
INPUT_FILE='/home/alan/Workspace/other/fastText/data/ag_news.train.tfrecords-1-of-1'

def test_parse_spec():
    fc = FeatureColumns(
        True,
        VOCAB_FILE,
        VOCAB_SIZE,
        10,
        10,
        1000,
        10)
    parse_spec = tf.feature_column.make_parse_example_spec(fc)
    print parse_spec
    reader = tf.python_io.tf_record_iterator(INPUT_FILE)
    sess = tf.Session()
    for record in reader:
        example = tf.parse_single_example(
            record,
            parse_spec)
        print sess.run(example)
        break


def test_reading_inputs():
    parse_spec = {
        "text": tf.VarLenFeature(tf.string),
        "label": tf.FixedLenFeature(shape=(1,), dtype=tf.int64,
                                    default_value=None)
    }
    sess = tf.Session()
    reader = tf.python_io.tf_record_iterator(INPUT_FILE)
    ESZ = 4
    HSZ = 100
    NC = 4
    n = 0
    text_lookup_table = tf.contrib.lookup.index_table_from_file(
        VOCAB_FILE, 10, VOCAB_SIZE)
    text_embedding_w = tf.Variable(tf.random_uniform(
        [VOCAB_SIZE, ESZ], -1.0, 1.0))
    sess.run([tf.tables_initializer()])
    for record in reader:
        example = tf.parse_single_example(
            record,
            parse_spec)
        labels = example["label"]
        labels1 = tf.subtract(labels, 1)
        dense = tf.sparse_tensor_to_dense(example["text"], default_value=" ")

        text_ids = text_lookup_table.lookup(dense)
        text_embedding = tf.reduce_mean(tf.nn.embedding_lookup(
            text_embedding_w, text_ids), axis=-2)
        print text_embedding.shape
        text_embedding = tf.expand_dims(text_embedding, -2)
        print text_embedding.shape
        num_classes = 2
        logits = tf.contrib.layers.fully_connected(
            inputs=text_embedding, num_outputs=4,
            activation_fn=None)
        sess.run([tf.global_variables_initializer()])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels1, logits=logits)
        print sess.run([text_embedding, logits, labels, loss])
        if n > 50:
            break
        n += 1
        # Test bow_encoder
        text_embedding_2 = tf.contrib.layers.bow_encoder(
            text_ids, VOCAB_SIZE, ESZ)
        print text_embedding_2.shape
        print sess.run(text_embedding_2)


if __name__ == '__main__':
    print "Test Parse Spec:"
    test_parse_spec()
    print "Test Input Fn"
    test_reading_inputs()
