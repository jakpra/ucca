#!/usr/bin/env bash

case "$TEST_SUITE" in
unit)
    # unit tests
    python -m unittest discover -v
    # basic conversion test
    ci/test_convert_toy.sh
    # basic parser tests
    python parsing/parse.py -I 10 -t doc/toy.xml -d doc/toy.xml -vm model_toy
    python parsing/parse.py doc/toy.xml -evm model_toy
    python parsing/parse.py -I 10 -t doc/toy.xml -d doc/toy.xml -svm model_toy_sentences
    python parsing/parse.py doc/toy.xml -esvm model_toy_sentences
    python parsing/parse.py -I 10 -t doc/toy.xml -d doc/toy.xml -avm model_toy_paragraphs
    python parsing/parse.py doc/toy.xml -esvm model_toy_paragraphs
    ;;
sparse)
    python parsing/parse.py -c sparse -WeLMCbs pickle/dev -t pickle/train
    ;;
dense)
    python parsing/parse.py -c dense -w word_vectors/sskip.100.vectors.txt -WeLMCbs pickle/dev -t pickle/train
    ;;
nn)
    python parsing/parse.py -c nn -w word_vectors/sskip.100.vectors.txt -WeLMCbs pickle/dev -t pickle/train
    ;;
tune)
    export W2V_FILE=word_vectors/sskip.100.vectors.txt
    python parsing/tune.py doc/toy.xml -t doc/toy.xml
    column -t -s, params.csv
    ;;
convert)
    ci/test_convert_all.sh
    ;;
convert_sentences)
    mkdir -p pickle/sentences
    python scripts/standard_to_sentences.py pickle/*.pickle -o pickle/sentences -p "ucca_passage" -b
    ci/test_convert_all_sentences.sh
    ;;
esac
