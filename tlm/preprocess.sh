TEXT=./data
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/train.txt \
    --validpref $TEXT/val.txt \
    --testpref $TEXT/test.txt \
    --destdir data-bin \
    --workers 1 &> log_preprocess.txt
