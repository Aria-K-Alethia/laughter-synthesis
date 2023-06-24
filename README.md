# Laughter synthesis using Pseudo Phonetic Tokens
Official implementation of the paper "Laughter Synthesis using Pseudo Phonetic Tokens with a Large-scale In-the-wild Laughter Corpus" accepted by *INTERSPEECH 2023*.

>**Abstract:**<br>
We present a large-scale in-the-wild Japanese laughter corpus and a laughter synthesis method.
Previous work on laughter synthesis lacks not only data but also proper ways to represent laughter.
To solve these problems, we first propose an in-the-wild corpus comprising $3.5$ hours of laughter, which is to our best knowledge the largest laughter corpus designed for laughter synthesis.
We then propose pseudo phonetic tokens (PPTs) to represent laughter by a sequence of discrete tokens, which are obtained by training a clustering model on features extracted from laughter by a pretrained self-supervised model.
Laughter can then be synthesized by feeding PPTs into a text-to-speech system.
We further show PPTs can be used to train a language model for unconditional laughter generation.
Results of comprehensive subjective and objective evaluations demonstrate that the proposed method significantly outperforms a baseline method, and can generate natural laughter unconditionally.

[[paper]](https://arxiv.org/abs/2305.12442)
[[demo]](https://aria-k-alethia.github.io/2023laughter-demo/)

# Setup
Please follow the steps below to prepare your environment and data:
- clone this repo
- `pip install -r requirements.txt`
- download vocoder [here](https://drive.google.com/file/d/1vvmqo0Aq0TGmAwfHuBqNudhzYf1UUQwu/view?usp=sharing) and put it under `hifigan` dir.
- download the proposed laughter corpus at [here](https://sites.google.com/site/shinnosuketakamichi/research-topics/laughter_corpus)

Then preprocess your data by:
```bash
python3 preprocess.py hydra.output_subdir=null hydra.job.chdir=False preprocess=laughter preprocess.path.laughter.path=[path to the corpus]
```
If everything goes well, you should find the processed data under `data/laughter`.

# Train
```bash
python3 train.py preprocess=laughter dataset=laughter
```
This will train the proposed TTS model using pseudo phonetic tokens as the representation of laughter with the default setting.

# Token language model
```bash
bash ./scripts/tlm.sh
```
This will train the proposed token language model with the default setting.

After training, you can sample new samples with `tlm/sample.sh` or evaluate the model with `tlm/eval.sh`.

# Citation
Please kindly cite the following paper if you find the corpus, code, or paper is helpful for your work:
```
@inproceedings{xin2023laughter
  title={Laughter Synthesis using Pseudo Phonetic Tokens with a Large-scale In-the-wild Laughter Corpus},
  author={Xin, Detai and Takamichi, Shinnosuke and Morimatsu, Ai and Saruwatari, Hiroshi},
  booktitle={Proc. Interspeech},
  year={2023}
}
```

# Acknowledgement
Part of the code in this repo is inspired by the following works:
- [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)
- [jik876/hifi-gan](https://github.com/jik876/hifi-gan)
- [fairseq/gslm](https://github.com/facebookresearch/fairseq/tree/main/examples/textless_nlp/gslm)

# Licence
MIT
