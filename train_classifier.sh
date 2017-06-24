#!/bin/bash

[ -d "$1" ] || {
    echo "Usage: train_classifer.sh data_dir [dataset_name=ag_news]"
    exit 1
}

set +v

DATADIR=$1
DATASET=${2:-ag_news}
OUTPUT=$DATADIR/models/${DATASET}
EXPORT_DIR=$DATADIR/models/${DATASET}
INPUT_TRAIN_FILE=$DATADIR/${DATASET}.train
INPUT_TEST_FILE=$DATADIR/${DATASET}.test
TRAIN_FILE=$DATADIR/${DATASET}.train.tfrecords-1-of-1
TRAIN_FILE=$DATADIR/${DATASET}.train.tfrecords-1-of-1
TEST_FILE=$DATADIR/${DATASET}.test.tfrecords-1-of-1

if [ ! -f ${TRAIN_FILE} ]; then
    echo Processing training dataset file
    python process_input.py --facebook_input=${INPUT_TRAIN_FILE} --ngrams=2,3,4
fi

if [ ! -f ${TEST_FILE} ]; then
    echo Processing test dataset file
    python process_input.py --facebook_input=${INPUT_TEST_FILE} --ngrams=2,3,4
fi

LABELS=$DATADIR/${DATASET}.train.labels
VOCAB=$DATADIR/${DATASET}.train.vocab
VOCAB_SIZE=`cat $VOCAB | wc -l | sed -e "s/[ \t]//g"`

echo $VOCAB
echo $VOCAB_SIZE

python classifier.py \
    --train_records=$TRAIN_FILE \
    --eval_records=$TEST_FILE \
    --label_file=$LABELS \
    --vocab_file=$VOCAB \
    --vocab_size=$VOCAB_SIZE \
    --num_oov_vocab_buckets=100 \
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

