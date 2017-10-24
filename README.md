# FastText in Tensorflow

This based on the ideas in Facebook's [FastText](https://github.com/facebookresearch/fastText) but implemented in
Tensorflow. However, it is not an exact replica of fastText.

Classification is done by embedding each word, taking the mean
embedding over the full text and classifying that using a linear
classifier. The embedding is trained with the classifier.  You can
also specify to use 2+ character ngrams. These ngrams get hashed then
embedded in a similar manner to the orginal words. Note, ngrams make
training much slower but only make marginal improvements in
performance, at least in English.

I may implement skipgram and cbow training later. Or preloading
embedding tables.

<< Still WIP >>

You can use [Horovod](https://github.com/uber/horovod) to distribute
training across multiple GPUs, on one or multiple servers. See usage
section below.

## FastText Language Identification

I have added utilities to train a classifier to detect languages, as
described in [Fast and Accurate Language Identification using
FastText](https://fasttext.cc/blog/2017/10/02/blog-post.html)

See usage below. It basically works in the same way as default usage.

## Implemented:
- classification of text using word embeddings
- char ngrams, hashed to n bins
- training and prediction program
- preprocess facebook format, or text input into tensorflow records

## Not Implemented:
- separate word vector training
- heirarchical softmax.
- quantize models

# Usage

The following are examples of how to use the applications. Get full help with
`--help` option on any of the programs.

To transform input data into tensorflow Example format, an example:

    process_input.py --facebook_input=queries.txt --output_dir=. --ngrams=2,3,4

Or, using a text file with one example per line with an extra file for labels:

    process_input.py --text_input=queries.txt --labels=labels.txt --output_dir=.

To train a text classifier:

    classifier.py \
      --train_records=queries.tfrecords \
      --eval_records=queries.tfrecords \
      --label_file=labels.txt \
      --vocab_file=vocab.txt \
      --model_dir=model \
      --export_dir=model

To predict classifications for text, use a saved_model from
classifier. `classifier.py --export_dir` stores a saved model in a
numbered directory below `export_dir`. Pass this directory to the
following to use that model for predictions:

    predictor.py
      --saved_model=model/12345678
      --text="some text to classify"
      --signature_def=proba

To export the embedding layer you can export from predictor. Note,
this will only be the text embedding, not the ngram embeddings.

    predictor.py
      --saved_model=model/12345678
      --text="some text to classify"
      --signature_def=embedding

Use the provided script to train easily:

    train_classifier.sh path-to-data-directory

# Language Identification

To implement something similar to the method described in [Fast and
Accurate Language Identification using
FastText](https://fasttext.cc/blog/2017/10/02/blog-post.html) you need to download the data:

    lang_dataset.sh [datadir]

You can then process the training and validation data using
`process_input.py` and `classifier.py` as described above.

There is a utility script to do this for you:

    python train_langdetect.py datadir


# Distributed Training

You can run training across multiple GPUs either on one or multiple
servers. To do so you need to install MPI and
[Horovod](https://github.com/uber/horovod) then add the `--horovod`
option. It runs very close to the GPU multiple in terms of
performance. I.e. if you have 2 GPUs on your server, it should run
close to 2x the speed.

    NUM_GPUS=2
    mpirun -np $NUM_GPUS python classifier.py \
      --horovod \
      --train_records=queries.tfrecords \
      --eval_records=queries.tfrecords \
      --label_file=labels.txt \
      --vocab_file=vocab.txt \
      --model_dir=model \
      --export_dir=model

The training script has this option added: `train_classifier.py`.

# Facebook Examples

<< NOT IMPLEMENTED YET >>

You can compare with Facebook's fastText by running similar examples
to what's provided in their repository.

    ./classification_example.sh
    ./classification_results.sh
