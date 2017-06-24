#!/bin/bash -v
# Usage: train_classiifer.sh data_dir

DATADIR=$1
OUTPUT=$DATADIR/model
EXPORT_DIR=$DATADIR/model
TRAIN_FILE=$DATADIR/ag_news.train.tfrecords-1-of-1
TEST_FILE=$DATADIR/ag_news.test.tfrecords-1-of-1
LABELS=$DATADIR/ag_news.train.labels
VOCAB=$DATADIR/ag_news.train.vocab
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

