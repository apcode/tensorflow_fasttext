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

## Implemented:
- classification of text using word embeddings
- char ngrams, hashed to n bins
- train and test program
- preprocess facebook format, or text input into tensorflow records

## Not Implemented:
- separate word vector training
- heirarchical softmax.
- quantize models

# Usage

The following are examples of how to use the applications. Get full help with
`--help` option on any of the programs.

To transform input data into tensorflow Example format, an example:

    process_input.py --facebook_input=queries.txt --model_dir=. --ngrams=2,3,4

Or, using a text file with one example per line with an extra file for labels:

    process_input.py --text_input=queries.txt --labels=labels.txt --model_dir=.

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

# Facebook Examples

<< NOT IMPLEMENTED YET >>

You can compare with Facebook's fastText by running similar examples
to what's provided in their repository.

    ./classification_example.sh
    ./classification_results.sh
