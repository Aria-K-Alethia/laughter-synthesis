'''
	Copyright (c) Xin Detai@University of Tokyo

	Description:
		speech2unit using SSL models
	Licence:
		MIT
	THE USER OF THIS CODE AGREES TO ASSUME ALL LIABILITY FOR THE USE OF THIS CODE.
	Any use of this code should display all the info above.
'''
import torch
import torch.nn as nn
import torchaudio
import os
import random
import argparse
import numpy as np
from os.path import join, exists, basename
from module.kmeans import MiniBatchKMeans
from module.ssl import SSLWrapper
from utils import load_audio, load_audio_with_resample, read_filelist, write_filelist
from tqdm import tqdm

seed = 1024

def parse_code(code):
    code = code.split()
    scode, length = [], []
    cur_code, cur_len = code[0], 0
    for c in code:
        if c == cur_code:
            cur_len += 1
        else:
            scode.append(cur_code)
            length.append(cur_len)
            cur_code = c
            cur_len = 1
    scode.append(cur_code)
    length.append(cur_len)
    assert len(scode) == len(length)
    return scode, length

def load_ssl_model(model_path):
    ssl_model = SSLWrapper(model_path)
    if torch.cuda.device_count() != 0:
        ssl_model = ssl_model.cuda()
    ssl_model.eval()
    return ssl_model

def ssl_features_generator(filelist, ssl_model, feature_type, model_path, layer):
    #root = os.path.dirname(filelist[0]).replace('wav', feature_type)
    root = join('./data/', feature_type, str(layer))
    os.makedirs(root, exist_ok=True)
    for filepath in tqdm(filelist):
        audio, _ = load_audio_with_resample(filepath, to_torch=True)
        feat_path = join(root, basename(filepath)[:-3] + 'pt')
        if not exists(feat_path):
            with torch.no_grad():
                if torch.cuda.device_count() != 0:
                    audio = audio.cuda()
                feat = ssl_model(audio) # [layer, B, T, F]
                feat = feat[layer-1].squeeze(0)
                feat = feat.cpu()
                torch.save(feat, feat_path)
                feat = feat.numpy()
        else:
            feat = torch.load(feat_path)
            feat = feat.numpy()
        yield feat

def get_ssl_features(filelist, ssl_model, feature_type, model_path, layer, pct=1.0, flatten=True):
    '''
        Extract SSL features for audio files

        Args:
            filelist: list, each item contain a wav file path
            feature_type: should be hubert, wav2vec2
            model_path: hugging face model path
            layer: extract feature from a specific layer
            pct: percentage of data that is extrcted
            flatten: whether to concat all features into a single tensor
        Returns:
            feats: list or np.array if flatten is set
    '''
    filelist = random.sample(filelist, int(len(filelist)*pct))
    print(f'Selected {len(filelist)} files, pct={pct}')
    generator = ssl_features_generator(filelist, ssl_model, feature_type, model_path, layer)
    feats = []
    for feat in generator:
        feats.append(feat)
    if flatten:
        feats = np.concatenate(feats, axis=0)
    return feats

def main(args):
    # laod ssl model
    print(f'Load ssl model {args.model_path}')
    ssl_model = load_ssl_model(args.model_path)
    if torch.cuda.device_count() != 0:
        ssl_model = ssl_model.cuda()
    ssl_model.eval()
    # load filelist
    train_filelist = [item[0] for item in read_filelist(args.train_filelist)]
    test_filelist = [item[0] for item in read_filelist(args.test_filelist)]
    print(f'Train: {len(train_filelist)}')
    # get ssl features if we need to train kmeans
    if args.pretrained_kmeans is None or not exists(args.pretrained_kmeans):
        feats = get_ssl_features(train_filelist, ssl_model,
                            args.feature_type,
                            args.model_path, args.layer,
                            pct=1.0, flatten=True) # [B, F]
    # train kmeans model or load pretrained model
    if args.pretrained_kmeans is None or not exists(args.pretrained_kmeans):
        kmeans = MiniBatchKMeans(args.nclusters, random_state=seed)
        print(kmeans)
        kmeans.fit(feats)
        print(f'Save kmeans model to {args.kmeans_path}')
        kmeans.save(args.kmeans_path)
        inertia = kmeans.compute_inertia(feats)
        print(f'Average inertia: {inertia}')
    else:
        print(f'Load pretrained kmeans model from {args.pretrained_kmeans}')
        kmeans = MiniBatchKMeans(args.nclusters, pretrained_path=args.pretrained_kmeans)
        print(kmeans)
    # quantize
    generator = ssl_features_generator(test_filelist, ssl_model,
                            args.feature_type, args.model_path,
                            args.layer)
    code_out = []
    print(f'Quantize test filelists, length: {len(test_filelist)}')
    for i, feat in enumerate(generator):
        pred = kmeans.predict(feat)
        pred_str = ' '.join(str(p) for p in pred)
        fname = basename(test_filelist[i])[:-4]
        code_out.append((fname, pred_str))
    # save code
    print(f'Write code output in {args.code_path}')
    write_filelist(code_out, args.code_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-filelist', type=str, default=None, help='filelist for kmeans training')
    parser.add_argument('--pretrained-kmeans', type=str, default=None)
    parser.add_argument('--nclusters', type=int, default=100, help='number of cluster in kmeans')
    parser.add_argument('--feature-type', type=str, default=None)
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--layer', type=int, default=-1)
    parser.add_argument('--test-filelist', type=str, default=None, help='filelist for code generation')
    parser.add_argument('--kmeans-path', type=str, default=None, help='kmeans model output path')
    parser.add_argument('--code-path', type=str, default=None, help='code output path')

    args = parser.parse_args()
    main(args)
