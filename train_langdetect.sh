#!/bin/bash

if [[ ! -d $1 ]]
then
    echo "Usage: train_langdetect.sh data_dir"
    exit 1
fi

if [[ ! -f $1/train.txt -o ! -f $1/valid.txt ]]
then
    echo "data_dir must contain train.txt and valid.txt from lang_dataset.sh"
    exit 2
fi

set +v

DATADIR=$1
OUTPUT=$DATADIR/models/${DATASET}
EXPORT_DIR=$DATADIR/models/${DATASET}
INPUT_TRAIN_FILE=$DATADIR/train.txt
INPUT_TEST_FILE=$DATADIR/valid.txt
TRAIN_FILE=$DATADIR/train.txt.tfrecords-1-of-1
TEST_FILE=$DATADIR/valid.txt.tfrecords-1-of-1

if [ ! -f ${TRAIN_FILE} ]; then
    echo Processing training dataset file
    python process_input.py --facebook_input=${INPUT_TRAIN_FILE} --ngrams=2,3,4
fi

if [ ! -f ${TEST_FILE} ]; then
    echo Processing test dataset file
    python process_input.py --facebook_input=${INPUT_TEST_FILE} --ngrams=2,3,4
fi

LABELS=$DATADIR/train.txt.labels
VOCAB=$DATADIR/train.txt.vocab
VOCAB_SIZE=`cat $VOCAB | wc -l | sed -e "s/[ \t]//g"`

echo $VOCAB
echo $VOCAB_SIZE
echo $LABELS

python classifier.py \
    --train_records=$TRAIN_FILE \
    --eval_records=$TEST_FILE \
    --label_file=$LABELS \
    --vocab_file=$VOCAB \
    --vocab_size=$VOCAB_SIZE \
    --model_dir=$OUTPUT \
    --export_dir=$EXPORT_DIR \
    --embedding_dimension=10 \
    --num_ngram_buckets=100000 \
    --ngram_embedding_dimension=10 \
    --learning_rate=0.01 \
    --batch_size=128 \
    --train_steps=1000 \
    --eval_steps=100 \
    --num_epochs=1 \
    --num_threads=1 \
    --nouse_ngrams \
    --nolog_device_placement \
    --fast \
    --debug
