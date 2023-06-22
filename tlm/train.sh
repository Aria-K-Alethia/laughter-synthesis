fairseq-train --task language_modeling \
  data-bin \
  --save-dir ckpts \
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.1 \
  --attention-dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --clip-norm 1.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode eos \
  --batch-size 16 --update-freq 2 \
  --num-workers 4 \
  --fp16 \
  --max-update 50000 \
  --log-interval 10 \
  --no-epoch-checkpoints \
  &> log_train.txt
