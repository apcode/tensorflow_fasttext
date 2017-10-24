#!/bin/bash

if [[ ! -d $1 ]]
then
    echo "Usage: train_langdetect.sh data_dir"
    exit 1
fi

if [[ ! -f $1/train.txt ]] || [[ ! -f $1/valid.txt ]]
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

echo "Looking for $TRAIN_FILE"
if ls ${TRAIN_FILE} 1> /dev/null 2>&1
then
    echo "Found"
else
    echo "Not Found $TRAIN_FILE"
    echo "Processing training dataset file"
    python process_input.py --facebook_input=${INPUT_TRAIN_FILE} --ngrams=2,3,4
    if ls ${TRAIN_FILE} 1> /dev/null 2>&1
    then
        echo "$TRAIN_FILE created"
    else
        echo "Failed to create $TRAIN_FILE"
        exit 1
    fi
fi

echo "Looking for $TEST_FILE"
if ls ${TEST_FILE} 1> /dev/null 2>&1
then
    echo "Found"
else
    echo "Not Found $TEST_FILE"
    echo "Processing test dataset file"
    python process_input.py --facebook_input=${INPUT_TEST_FILE} --ngrams=2,3,4
    if ls ${TEST_FILE} 1> /dev/null 2>&1
    then
        echo "$TEST_FILE created"
    else
        echo "Failed to create $TEST_FILE"
        exit 1
    fi
fi

LABELS=$DATADIR/train.txt.labels
VOCAB=$DATADIR/train.txt.vocab
VOCAB_SIZE=`cat $VOCAB | wc -l | sed -e "s/[ \t]//g"`

echo $VOCAB
echo $VOCAB_SIZE
echo $LABELS

# python classifier.py \
#     --train_records=$TRAIN_FILE \
#     --eval_records=$TEST_FILE \
#     --label_file=$LABELS \
#     --vocab_file=$VOCAB \
#     --vocab_size=$VOCAB_SIZE \
#     --model_dir=$OUTPUT \
#     --export_dir=$EXPORT_DIR \
#     --embedding_dimension=10 \
#     --num_ngram_buckets=100000 \
#     --ngram_embedding_dimension=10 \
#     --learning_rate=0.01 \
#     --batch_size=128 \
#     --train_steps=1000 \
#     --eval_steps=100 \
#     --num_epochs=1 \
#     --num_threads=1 \
#     --nouse_ngrams \
#     --nolog_device_placement \
#     --fast \
#     --debug

mpirun -np 2 python classifier.py \
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
    --horovod \
    --debug
