fairseq-eval-lm data-bin \
    --path ckpts/checkpoint_best.pt \
    --batch-size 12 \
    --tokens-per-sample 512 --sample-break-mode eos \
    --context-window 0 &> log_eval.txt
