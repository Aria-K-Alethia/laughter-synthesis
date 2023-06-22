import os
import hydra
import librosa
import soundfile as sf
import utils
import numpy as np
import subprocess
import torch
import random
from os.path import join, exists, basename
from glob import glob
from collections import defaultdict
from utils import read_filelist, write_filelist
from dataset import Laughter
from tqdm import tqdm
from speech2unit import parse_code
from preprocessor import Preprocessor

@hydra.main(version_base=None, config_path='config', config_name='default')
def preprocess(cfg):
    cfg = cfg.preprocess
    filelist = []
    # laughter
    laughter = Laughter(cfg.path.laughter.path)
    laughter_wavs = laughter.collect_files()
    filelist.extend(laughter_wavs)
    print(f'Collected {len(filelist)} files')
    
    # resample the audio to 16 khz
    os.makedirs('data', exist_ok=True)
    tgt_dir = join(cfg.path.processed_path)
    os.makedirs(tgt_dir, exist_ok=True)
    tgt_wav_dir = join(tgt_dir, 'wav')
    os.makedirs(tgt_wav_dir, exist_ok=True)
    fids = []
    for f in tqdm(laughter_wavs):
        fid = basename(f)[:-4]
        fids.append(fid)
        tgt_wav_path = join(tgt_wav_dir, fid+'.wav')
        if exists(tgt_wav_path):
            continue
        src_wav_path = f
        wav, sr = utils.read_audio(src_wav_path)
        if len(wav.shape) == 2:
            wav = wav[:, 0]
        new_wav = wav if sr == cfg.audio.sr else librosa.resample(wav, orig_sr=sr, target_sr=cfg.audio.sr)
        # trim silence
        new_wav, _ = librosa.effects.trim(new_wav, cfg.audio.top_db, frame_length=cfg.stft.window_length, hop_length=cfg.stft.hop_length)
        new_wav = new_wav / max(abs(new_wav)) * cfg.audio.max_wav_value
        utils.write_audio(new_wav.astype(np.int16), tgt_wav_path, cfg.audio.sr, 'PCM_16')
    
    # kmeans filelist
    if not exists(cfg.view.kmeans_filelist):
        kmeans_files = []
        for fid in fids:
            path = join(tgt_wav_dir, fid + '.wav')
            kmeans_files.append((path,))
        write_filelist(kmeans_files, cfg.view.kmeans_filelist)

    # generate code
    os.makedirs('ckpt', exist_ok=True)
    os.makedirs('codes', exist_ok=True)
    cmd = f'CUDA_VISIBLE_DEVICES=1 python3 speech2unit.py --train-filelist {cfg.view.kmeans_filelist} --nclusters {cfg.code.nclusters} --feature-type hubert --model-path facebook/hubert-base-ls960 --layer {cfg.code.layer} --test-filelist {cfg.view.kmeans_filelist} --kmeans-path {cfg.code.model_path} --code-path {cfg.code.code_path} --pretrained-kmeans {cfg.code.model_path}'
    print(cmd)
    if not exists(cfg.code.code_path):
        subprocess.run(cmd, shell=True)
    
    # code and length
    print(f'Dump code and duration')
    codes = read_filelist(cfg.code.code_path)
    tgt_code_dir = join(tgt_dir, 'code')
    tgt_len_dir = join(tgt_dir, 'duration')
    os.makedirs(tgt_code_dir, exist_ok=True)
    os.makedirs(tgt_len_dir, exist_ok=True)
    for fid, code in tqdm(codes):
        tgt_code_path, tgt_len_path = join(tgt_code_dir, fid+'.txt'), join(tgt_len_dir, fid+'.pt')
        if exists(tgt_code_path) and exists(tgt_len_path):
            continue
        code, length = parse_code(code)
        with open(tgt_code_path, 'w') as f:
            f.write(' '.join(code))
        length = np.array(length)
        torch.save(length, tgt_len_path)
    # speaker
    tgt_spkr_path = join(tgt_dir, 'speaker.pt')
    print(f'Dump speaker dict to {tgt_spkr_path}')
    if not exists(tgt_spkr_path):
        speakers = list(set(list(laughter.collect_speakers())))
        speakers.sort()
        speakers = {s: i for i, s in enumerate(speakers)}
        torch.save(speakers, tgt_spkr_path)
    else:
        speakers = torch.load(tgt_spkr_path)
    
    tgt_spkr_dir = join(tgt_dir, 'speaker')
    os.makedirs(tgt_spkr_dir, exist_ok=True)
    for wav in laughter_wavs:
        fid = basename(wav)[:-4]
        tgt_spkr_path = join(tgt_spkr_dir, fid+'.txt')
        spkr = laughter.get_speaker(fid)
        with open(tgt_spkr_path, 'w') as f:
            f.write(spkr)
    print(f'total speaker number: {len(speakers)}')
    
    # mel, f0
    print(f'Dump acoustic features')
    preprocessor = Preprocessor(cfg)
    processed_fids = preprocessor.process(tgt_dir, fids)
    print(f'preprocess done, before: {len(fids)}, after: {len(processed_fids)}')    
    
    # train filelist
    train_filelist, val_filelist, test_filelist = [], [], []
    laughter_fids = [basename(wav)[:-4] for wav in laughter_wavs]
    laughter_fids = [fid for fid in laughter_fids if fid in processed_fids]
    spkr2fids = defaultdict(list)
    _ = [spkr2fids[laughter.get_speaker(fid)].append((fid,)) for fid in laughter_fids]
    spkr2fids = sorted(list(spkr2fids.items()), key=lambda x: -len(x[1]))
    print([(k, len(v)) for k, v in spkr2fids])
    n_test_spkr = 32
    n_val_spkr = 128
    n = 5
    for i, (spkr, fids) in enumerate(spkr2fids):
        random.shuffle(fids)
        train, val, test = [], [], []
        if i < n_test_spkr and len(fids) > n:
            fids, test = fids[:-n], fids[-n:]
        if i < n_val_spkr and len(fids) > 1:
            fids, val = fids[:-1], fids[-1:]
        train = fids
        train_filelist.extend(train)
        val_filelist.extend(val)
        test_filelist.extend(test)
    print(f'train: {len(train_filelist)}, val: {len(val_filelist)}, test: {len(test_filelist)}')
    if not exists(cfg.view.train_filelist):
        write_filelist(train_filelist, cfg.view.train_filelist)
    if not exists(cfg.view.val_filelist):
        write_filelist(val_filelist, cfg.view.val_filelist)
    if not exists(cfg.view.test_filelist):
        write_filelist(test_filelist, cfg.view.test_filelist)
    

if __name__ == '__main__':
    preprocess()
