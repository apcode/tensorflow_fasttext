# FastText in Tensorflow

This is a version of Facebook's [FastText](https://github.com/facebookresearch/fastText) implemented in
Tensorflow. However, it is not an exact replica of fastText. Instead I
implement the main classifier using either a skipgram or cbow word
vector embedding. I do not intend to implement heirarchical
softmax, instead using nce loss for training.

## Implemented:
- skipgram and cbow word and sentence vectors
- out of vocab word vectors 
- classification of text
- char ngrams, hashed to vectors (not yet implemented)
- train, test and predict programs

## Not Implemented:
- heirarchical softmax (I'm using NCE (sampled) loss instead).
- quantize models

# Usage

The following are examples of how to use the applications. Get full help with
`--help` option on any of the programs.

To transform input data into tensorflow Example format, an example:

    process_input.py --facebook_input=queries.txt --output_dir=.

Or, using a text file with one example per line with an extra file for labels:

    process_input.py --text_input=queries.txt --labels=labels.txt --output_dir=.

To train a new skipgram word vector model:

    train_vectors.py --skipgram --input=queries.tfrecords --output_dir=model
    
To train a text classifier:

    train_classifier.py --input=queries.tfrecords --output_dir=model
    
# Facebook Examples

You can compare with Facebook's fastText by running similar examples
to what's provided in their repository.

    ./classification_example.sh
    ./classification_results.sh
    ./word_vector_example.sh

