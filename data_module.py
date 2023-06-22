import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import pytorch_lightning as pl
import random
from os.path import basename, exists, join
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from functools import partial
import hydra
import utils

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        ocwd = hydra.utils.get_original_cwd()
        self.ocwd = ocwd

    def get_loader(self, phase):
        phase_cfg = self.cfg.dataset.get(phase)
        batch_size = phase_cfg.batch_size
        #ds = hydra.utils.instantiate(self.cfg.dataset.dataset, phase, self.cfg)
        ds = FSDataset(phase, self.cfg)
        '''
        if phase == 'train':
            sampler = CustomSampler(ds, True)
        else:
            sampler = None
        '''
        sampler = None
        dl = DataLoader(ds, batch_size, phase_cfg.shuffle, num_workers=8, sampler=sampler, collate_fn=ds.collate_fn)

        return dl

    def train_dataloader(self):
        return self.get_loader('train')

    def val_dataloader(self):
        return self.get_loader('val')

    def test_dataloader(self):
        return self.get_loader('test')

class FSDataset(Dataset):
    """FastSpeech dataset batching text, mel, pitch 
    and other acoustic features

    Args:
        phase: train, val, test
        cfg: hydra config
    """
    def __init__(self, phase, cfg):
        self.phase = phase
        self.cfg = cfg
        self.phase_cfg =  cfg.dataset.get(phase)
        self.ocwd = hydra.utils.get_original_cwd()
        self.feat_path = join(self.ocwd, cfg.preprocess.path.processed_path)
        self.sr = cfg.preprocess.audio.sr
        
        self.filelist = utils.read_filelist(join(self.ocwd, self.phase_cfg.filelist))
        self.speaker_dict = torch.load(join(self.feat_path, 'speaker.pt'))
    
    def __len__(self):
        return len(self.filelist)

    def parse_fid(self, fid):
        fid = fid.split('-')
        speaker = ''.join(fid[:-1])
        num = fid[-1]
        return speaker, num

    def load_text(self, fid, name):
        path = join(self.feat_path, name, utils.text_suffix(fid))
        with open(path) as f:
            text = f.read().strip()
        return text

    def load_wav(self, fid):
        path = join(self.feat_path, 'wav', utils.wav_suffix(fid))
        wav, sr = utils.read_audio(path)
        return wav

    def load_feature(self, fid, name):
        path = join(self.feat_path, name, utils.torch_suffix(fid))
        feat = torch.load(path)
        return feat
    
    def __getitem__(self, idx):
        (fid,) = self.filelist[idx]
        speaker = self.load_text(fid, 'speaker')
        speaker_id = self.speaker_dict[speaker]
        raw_phone = list(map(int, self.load_text(fid, 'code').split()))
        phone = torch.LongTensor([s+1 for s in raw_phone]) # add one to leave 0 as pad token
        wav = self.load_wav(fid)
        
        mel = torch.from_numpy(self.load_feature(fid, 'mel')).float() # [#mel, Tm]
        pitch = torch.from_numpy(self.load_feature(fid, 'pitch')).float() # [Tt] or [Tm]
        energy = torch.from_numpy(self.load_feature(fid, 'energy')).float() # [Tt] or [Tm]
        duration = torch.LongTensor(self.load_feature(fid, 'duration')) # [Tt]
        
        out = {
            'fid': fid,
            'speaker': speaker_id,
            'raw_speaker': speaker,
            'text': phone,
            'raw_phone': raw_phone,
            'raw_text': raw_phone,
            'wav': wav,
            'mel': mel,
            'pitch': pitch,
            'energy': energy,
            'duration': duration,
        }
        
        return out
    
    def collate_fn(self, bs):
        indices = range(len(bs))
        def rearange(batch, feat):
            return [batch[i][feat] for i in indices]
        fids = rearange(bs, 'fid')
        speakers = rearange(bs, 'speaker')
        raw_speakers = rearange(bs, 'raw_speaker')
        texts = rearange(bs, 'text')
        raw_phones = rearange(bs, 'raw_phone')
        raw_texts = rearange(bs, 'raw_text')
        wavs = rearange(bs, 'wav')
        mels = rearange(bs, 'mel')
        pitches = rearange(bs, 'pitch')
        energies = rearange(bs, 'energy')
        durations = rearange(bs, 'duration')

        text_lengths = torch.LongTensor([len(text) for text in texts])
        mel_lengths = torch.LongTensor([mel.shape[-1] for mel in mels])
        
        text_max_length = text_lengths.max().item()
        mel_max_length = mel_lengths.max().item()
        
        speakers = torch.LongTensor(speakers)
        texts = torch.stack(utils.pad_torch(texts))
        mels = torch.stack(utils.pad_torch(mels))
        pitches = torch.stack(utils.pad_torch(pitches))
        energies = torch.stack(utils.pad_torch(energies))
        durations = torch.stack(utils.pad_torch(durations))

        out = {
            'fid': fids,
            'speaker': speakers,
            'raw_speaker': raw_speakers,
            'text': texts,
            'raw_phone': raw_phones,
            'raw_text': raw_texts,
            'wav': wavs,
            'mel': mels,
            'pitch': pitches,
            'energy': energies,
            'duration': durations,
            'text_length': text_lengths,
            'mel_length': mel_lengths,
            'text_max_length': text_max_length,
            'mel_max_length': mel_max_length,
        }
        return out
        
class CustomSampler(Sampler):
    def __init__(self, ds, shuffle):
        self.ds = ds
        self.shuffle = shuffle
        self.verbal_indices = []
        self.nv_indices = []
        self.nv_spkrs = ['matsumoto', 'yamamoto', 'yokoda', 'nagashima']
        self.n = 10
        for i, (item, ) in enumerate(ds.filelist):
            spkr = item.split('_')[0]
            if spkr in self.nv_spkrs:
                self.nv_indices.append(i)
            else:
                self.verbal_indices.append(i)

    def __iter__(self):
        indices = self.verbal_indices + self.nv_indices * self.n
        if self.shuffle:
            random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.verbal_indices) + len(self.nv_indices) * self.n
