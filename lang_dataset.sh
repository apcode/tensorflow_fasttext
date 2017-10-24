#!/bin/bash

if [[ $# -ne 1 ]]
then
    DATADIR="/tmp/lang_detect"
fi

if [[ ! -d $DATADIR ]]
then
    mkdir -p $DATADIR
    if [[ $? -ne 0 ]]
    then
        echo "Failed to create $DATADIR"
        echo "Usage: lang_dataset.sh [datadir]"
        exit 1
    fi
fi

set -v

pushd $DATADIR
wget http://downloads.tatoeba.org/exports/sentences.tar.bz2
bunzip2 sentences.tar.bz2
tar xvf sentences.tar
awk -F"\t" '{print"__label__"$2" "$3}' < sentences.csv | shuf > processed_sentences.txt
head -n 10000 processed_sentences.txt > valid.txt
tail -n +10001 processed_sentences.txt > train.txt
popd

ls -lh $DATADIR
echo DONE
