import os
import random
import librosa
import soundfile as sf
import json
import torch
import torchaudio
import numpy as np
import utils
from os.path import join, basename, exists, dirname
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from functools import partial
from collections import defaultdict
from hifigan import meldataset

class Preprocessor:
    """Preprocessor to get acoustic features for FastSpeech2
    """
    default_speaker = 'default'
    def __init__(self, cfg):
        self.cfg = cfg
        self.sr = cfg.audio.sr
        self.hop_length = cfg.stft.hop_length
        
        self.pitch_phoneme_averaging = (cfg.pitch.feature == 'phoneme')
        self.energy_phoneme_averaging = (cfg.energy.feature == 'phoneme')
        
        self.pitch_normalization = cfg.pitch.normalization
        self.energy_normalization = cfg.energy.normalization
        
        self.spec_module = torchaudio.transforms.Spectrogram(
            n_fft=cfg.stft.n_fft,
            win_length=cfg.stft.window_length,
            hop_length=cfg.stft.hop_length,
            power=1,
            center=True,
        )
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=cfg.mel.n_mel,
            sample_rate=cfg.audio.sr,
            f_min=cfg.mel.fmin,
            f_max=cfg.mel.fmax,
            n_stft=cfg.stft.n_fft // 2 + 1,
            norm="slaney",
            mel_scale="slaney",
        ).double()

        
    def process_utterance(self, root, fid):
        """Process a single utterance, generate mel, pitch, energy, duration
        This method will save the results to target dir
        Args:
            root (string): data root, should contain wav and tg dir to access wav and textgrid files
            text: text object containing all info for this text
        Returns:
            (fid, pitch, energy, mel_length)
        """
        duration_path = join(root, 'duration', fid+'.pt')
        pitch_path = join(root, 'pitch', fid+'.pt')
        energy_path = join(root, 'energy', fid+'.pt')
        mel_path = join(root, 'mel', fid+'.pt')
        # paths for wav, textgrid
        wav_path = join(root, 'wav', fid+'.wav')
        # get alignment info from textgrid
        duration = torch.load(duration_path)
        nframe = sum(duration)
        if nframe > 1000:
            print(f'{fid}, length: {nframe}, too long')
            return (fid, None, None, None)
        # load wav
        wav, local_sr = utils.read_audio(wav_path)
        assert local_sr == self.sr, f"{fid} has {local_sr} sr but global sr is {self.sr}"
        if abs(round(wav.shape[0] / self.hop_length)- sum(duration)) >= 3:
            print(f'{fid} has {round(wav.shape[0] / self.hop_length)} but the lab indicates it has {sum(duration)} frames')
            return (fid, None, None, None)
        # pitch
        pitch = utils.compute_robust_pitch(wav, self.sr, self.hop_length)
        if pitch is None:
            print(f'{fid} has no pitch')
            return (fid, None, None, None)
        pitch = pitch[: nframe]
        if np.sum(pitch != 0) <= 1:
            print(f'{fid} has no pitch')
            return (fid, None, None, None)
        # mel-spec
        mag_spec, mel_spec = self.compute_hifigan_spec(wav)
        mel_spec = mel_spec[:, : nframe]
        # energy
        energy = utils.compute_energy(mag_spec)
        energy = energy[:nframe]
        # process phoneme averaging
        if self.pitch_phoneme_averaging:
            pitch = self.get_phoneme_pitch(pitch, duration)
        if self.energy_phoneme_averaging:
            energy = self.get_phoneme_energy(energy, duration)
        # dump files, assume the target dir exists
        torch.save(pitch, pitch_path)
        torch.save(energy, energy_path)
        torch.save(mel_spec, mel_path)
        return fid, pitch, energy, mel_spec.shape[-1]
        
    def process(self, root, fids):
        """process a batch of fids in root dir

        Args:
            root (string): root dir, should contain wav, text, tg
            texts: processed text objects
        """
        print(f'Process {len(fids)} under {root} to get mel, pitch, energy and duration')
        # create dir
        os.makedirs(join(root, 'mel'), exist_ok=True)
        os.makedirs(join(root, 'pitch'), exist_ok=True)
        os.makedirs(join(root, 'energy'), exist_ok=True)
        # init, scaler
        n_frames = 0
        # per speaker normalization if set
        pitch_scaler = defaultdict(StandardScaler)
        energy_scaler = defaultdict(StandardScaler)
        
        # processing
        print('Process the utterances')
        outs = []
        for fid in tqdm(fids):
            out = self.process_utterance(root, fid)
            outs.append(out)
        print(f'Done, got {len(outs)} results, begin to collect statistics')
        failed, failed_list, processed_fids = 0, [], []
        for fid, pitch, energy, n in outs:
            if pitch is None or energy is None:
                failed += 1
                failed_list.append(fid)
                continue
            try:
                spkr = get_spkr_from_fid(root, fid)
                spkr = spkr if self.cfg.per_speaker_normalization else self.default_speaker
                fixed_pitch, fixed_energy = pitch, energy
                if fixed_pitch.size != 0:
                    pitch_scaler[spkr].partial_fit(fixed_pitch.reshape(-1, 1))
                if fixed_energy.size != 0:
                    energy_scaler[spkr].partial_fit(fixed_energy.reshape(-1, 1))
                processed_fids.append(fid)
                n_frames += n
            except Exception as e:
                print(f'Scaler partial fit {fid} failed {e}')
        print(f'Done, {failed} failed, failed list: {failed_list}')
        print(f'{len(processed_fids)} fid succeeded')
                
        # normalization
        ## pitch
        if self.pitch_normalization:
            pitch_mean = {k: v.mean_[0] for k, v in pitch_scaler.items()}
            pitch_std = {k: v.scale_[0] for k, v in pitch_scaler.items()}
        else:
            pitch_mean = {self.default_speaker: 0}
            pitch_std = {self.default_speaker: 1}
        ## energy
        if self.energy_normalization:
            energy_mean = {k: v.mean_[0] for k, v in energy_scaler.items()}
            energy_std = {k: v.scale_[0] for k, v in energy_scaler.items()}
        else:
            energy_mean = {self.default_speaker: 0}
            energy_std = {self.default_speaker: 1}
        #pitch_task = partial(self.normalize, join(root, 'pitch'))
        print(f'Pitch normalization, mean: {pitch_mean}, std: {pitch_std}')
        pitch_min, pitch_max = self.normalize(root, 'pitch', processed_fids, pitch_mean, pitch_std, self.cfg.pitch.norm_method)
        print(f'Pitch range: [{pitch_min:.3f}, {pitch_max:.3f}]')
        print(f'Energy normalization, mean: {energy_mean}, std: {energy_std}')
        energy_min, energy_max = self.normalize(root, 'energy', processed_fids, energy_mean, energy_std, self.cfg.energy.norm_method)
        print(f'Energy range: [{energy_min:.3f}, {energy_max:.3f}]')
        # save statistics of acoustic features
        stats_path = join(root, 'stats.pt')
        print(f'Save statistics to {stats_path}')
        stats = {
            'pitch': {
                'min': float(pitch_min),
                'max': float(pitch_max),
                'mean': pitch_mean,
                'std': pitch_std
            },
            'energy': {
                'min': float(energy_min),
                'max': float(energy_max),
                'mean': energy_mean,
                'std': energy_std
            },
        }
        torch.save(stats, stats_path)
        # done
        print(f'Process done, total time: {n_frames * self.hop_length / self.sr / 3600:.2f} hours')
        
        return processed_fids
        
    def normalize(self, root, feat, fids, mean, std, method):
        """normalize features under dirname using z-score
        This assume the file suffix is .pt

        Args:
            dirname (string): should be the dir saving acoustic feat
            fids: fid list
            mean, std: statistics
            method: support z_score, mean, none
        """
        dirname = join(root, feat)
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        
        for fid in fids:
            path = join(dirname, fid+'.pt')
            feat = torch.load(path)
            spkr = get_spkr_from_fid(root, fid) if self.cfg.per_speaker_normalization else self.default_speaker
            if method == 'z_score':
                feat = (feat - mean[spkr]) / std[spkr]
            elif method == 'z_score_utt':
                feat = (feat - feat.mean()) / feat.std()
            elif method == 'mean':
                feat = feat - mean[spkr]
            elif method == 'none':
                pass
            else:
                raise ValueError(f'No such normalization method: {method}')
            torch.save(feat, path)

            max_value = max(max_value, max(feat))
            min_value = min(min_value, min(feat))
        
        return min_value, max_value
    
    def compute_spec(self, audio):
        audio = torch.clip(torch.from_numpy(audio), -1, 1)
        magspec = self.spec_module(audio)
        #print(magspec, magspec.dtype, audio, audio.dtype)
        melspec = self.mel_scale(magspec)
        logmelspec = torch.log(torch.clamp_min(melspec, 1.0e-5) * 1.0).to(torch.float32)
        return magspec.numpy(), logmelspec.numpy()

    def compute_hifigan_spec(self, audio):
        audio = torch.clip(torch.from_numpy(audio), -1, 1)
        magspec, melspec = meldataset.mel_spectrogram(audio.float().unsqueeze(0), self.cfg.stft.n_fft, self.cfg.mel.n_mel, self.cfg.audio.sr, self.cfg.stft.hop_length, self.cfg.stft.window_length, self.cfg.mel.fmin, self.cfg.mel.fmax)
        return magspec.squeeze(0).numpy(), melspec.squeeze(0).numpy()
    
    def interpolate_pitch(self, pitch):
        # interpolate the pitch
        nonzero_ids = np.where(pitch != 0)[0]
        interp_fn = interp1d(
            nonzero_ids,
            pitch[nonzero_ids],
            fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
            bounds_error=False,    
        )
        pitch = interp_fn(np.arange(0, len(pitch)))
        return pitch

    def get_phoneme_pitch(self, pitch, duration):
        """Convert frame-level pitch into phoneme-level based on duration

        Args:
            pitch (np.array): frame level pitch
            duration (list): phoneme duration list

        Returns:
            pitch: phoneme level pitch
        """
        # phoneme averaging
        pos = 0
        for i, d in enumerate(duration):
            if d > 0:
                pitch[i] = np.mean(pitch[pos:pos+d])
            else:
                pitch[i] = 0
            pos += d
        pitch = pitch[:len(duration)]
        return pitch
        
    def get_phoneme_energy(self, energy, duration):
        pos = 0
        for i, d in enumerate(duration):
            if d > 0:
                energy[i] = np.mean(energy[pos:pos+d])
            else:
                energy[i] = 0
            pos += d
        energy = energy[:len(duration)]
        return energy            

def remove_outlier(feat):
    feat = np.array(feat)
    p25 = np.percentile(feat, 25)
    p75 = np.percentile(feat, 75)
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    normal_indices = np.logical_and(feat > lower, feat < upper)
    return feat[normal_indices]


def get_spkr_from_fid(root, fid):
    spkr_path = join(root, 'speaker', fid+'.txt')
    with open(spkr_path) as f:
        spkr = f.read().strip()
    return spkr
