import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import transformers
import soundfile as sf
import pyworld as pw
import numpy as np
import hifigan
import librosa
from os.path import join, exists, basename
import json
from scipy.interpolate import interp1d

def read_filelist(path, delimiter='|'):
    with open(path) as f:
        lines = [line.strip().split(delimiter) for line in f if line.strip()]
    return lines

def write_filelist(filelists, path, delimiter='|'):
    with open(path, 'w', encoding='utf8') as f:
        for line in filelists:
            f.write(delimiter.join(line) + '\n')

def load_ssl_model(model_path):
    model = transformers.HubertModel.from_pretrained(model_path)
    return model

def read_audio(path):
    wav, sr = sf.read(path)
    return wav, sr

def write_audio(audio, path, sr, subtype='PCM_16'):
    sf.write(path, audio, sr, subtype)

def load_audio(path, to_torch=False):
    wav, sr = sf.read(path)
    if len(wav.shape) == 1:
        wav = wav[None, :]
    if to_torch:
        wav = torch.from_numpy(wav).float()
    return wav, sr

def load_audio_with_resample(path, to_torch=False, target_sr=16000):
    wav, sr = sf.read(path)
    if target_sr != sr:
        wav = librosa.resample(wav, sr, target_sr)
    if len(wav.shape) == 1:
        wav = wav[None, :]
    if to_torch:
        wav = torch.from_numpy(wav).float()
    return wav, sr

def compute_pitch(wav, sr, hop_length):
    pitch, t = pw.dio(wav.astype(np.float64), sr, frame_period=hop_length / sr * 1000)
    pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sr)
    
    return pitch

def pitch_world(wav, sr, hop_length):
    pitch, t = pw.dio(wav.astype(np.float64), sr, frame_period=hop_length / sr * 1000, f0_ceil=800, allowed_range=0.2)
    pitch = pw.stonemask(wav.astype(np.float64), pitch, t, sr)
    
    return pitch

def downsample(pitch, down_rate):
    nonzero_ids = np.where(pitch != 0)[0]
    interp_fn = interp1d(
        nonzero_ids,
        pitch[nonzero_ids],
        fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
        bounds_error=False,    
    )
    pad = 0 if len(pitch) % down_rate == 0 else down_rate - (len(pitch) % down_rate)
    pitch = interp_fn(np.arange(0, len(pitch) + pad))
    nframe = pos = 0
    for i in range(len(pitch) // down_rate):
        pitch[i] = np.mean(pitch[pos:pos+down_rate])
        pos += down_rate
    pitch = pitch[:i]
    return pitch

def compute_robust_pitch(wav, sr, hop_length):
    hops = [320, 160, 80, 40, 20]
    assert hop_length == 320, f"hop_length {hop_length} should be 320 in this method"
    for hop in hops:
        pitch = pitch_world(wav, sr, hop)
        if np.sum(pitch != 0 ) <= 1:
            #print(f'Fail to extract f0 for {wavfile} with hop {hop}')
            continue
        break
    else:
        return None
    pitch = downsample(pitch, 320//hop)
    return pitch

def compute_energy(spec):
    # spec: numpy array [#F, T]
    energy = np.linalg.norm(spec, axis=0)
    return energy

def wav_suffix(s):
    return s+'.wav'

def text_suffix(s):
    return s+'.txt'

def torch_suffix(s):
    return s+'.pt'

def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask

def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded

def pad_torch(bs, max_len=None, value=0):
    """Pad torch tensor in the last dim

    Args:
        bs (iterable): data batch
        max_len (int, optional): maximal length of bs. Defaults to None.
        value (int, optional): pad value. Defaults to 0.

    Returns:
        out (list): pad results
    """
    if max_len is None:
        max_len = max(b.shape[-1] for b in bs)
    out = [
        F.pad(b, (0, max_len - b.shape[-1]), value=value)
        for b in bs
    ]
    return out

def get_vocoder_16k(name, path):
    print(f'Load vocoder: {name}')
    with open(join(path, "config_16k_320hop.json"), "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    ckpt = torch.load(join(path, "g_16k_320hop"), map_location=lambda s, l: s)
    vocoder.load_state_dict(ckpt["generator"], strict=True)
    vocoder.eval()
    vocoder.remove_weight_norm()
    return vocoder

def get_transformer_scheduler(optimizer, warmup_steps, d_model, last_epoch=-1, verbose=True):
    def lr_lambda(step):
        step += 1
        arg1 = step**-0.5
        arg2 = step*(warmup_steps**-1.5)
        lr = (d_model**-0.5) * min(arg1, arg2)
        return lr
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch, verbose=verbose)

def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)
