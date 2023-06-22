python3 sample.py data-bin/ \
	--path=ckpts/checkpoint_best.pt --task=language_modeling --sampling --temperature=0.7 \
	--seed=12345678  --prompts=data/prompt.txt  --output=data/sample.txt --max-len-a=0 --max-len-b=500 \
	--prefix-size=${1} --batch-size=16 --fp16 --samples-per-prompt=1
