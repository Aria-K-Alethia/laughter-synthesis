import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from module.transformer import Encoder, Decoder
from module.module import PostNet
from module.variance_adaptor import VarianceAdaptor
from utils import get_mask_from_lengths
from os.path import join

def model_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight, torch.nn.init.calculate_gain("relu"))

class FastSpeech2(nn.Module):
    """FastSpeech2 for non-augoregressive TTS

    Args:
        cfg: hydra config
    """

    def __init__(self, cfg):
        super(FastSpeech2, self).__init__()
        self.cfg = cfg
        self.model_cfg = cfg.model

        self.encoder = Encoder(cfg)
        self.variance_adaptor = VarianceAdaptor(cfg)
        self.decoder = Decoder(cfg)
        self.mel_linear = nn.Linear(
            cfg.model.transformer.decoder_hidden,
            cfg.preprocess.mel.n_mel
        )
            
        self.postnet = PostNet()

        self.speaker_emb = None
        ocwd = hydra.utils.get_original_cwd()
        if cfg.model.multi_speaker:
            speaker_dict = torch.load(join(ocwd, cfg.preprocess.path.processed_path, 'speaker.pt'))
            self.n_speaker = len(speaker_dict)
            self.speaker_emb = nn.Embedding(self.n_speaker, cfg.model.transformer.encoder_hidden)
        self.register_buffer('step', torch.LongTensor([-1]).squeeze())
        self.apply(model_init) 
    
    def increment_step(self):
        self.step += 1

    def forward(self, batch):
        """Forward method for FastSpeech2

        Args:
            batch: dict should contain all necessary data for training
            speaker: [B]
            text: [B, T]
            text_length: [B]
            text_max_length: int
            mel: [B, nmel, T]
            mel_length: [B]
            mel_max_length: int
            pitch: [B, T]
            energy: [B, T]
            duration: [B, T]
            p/e/d_control: float, controlling the pitch, energy and duration during inference
        Returns:
            mel_pred: [B, F, T]
            mel_postnet_pred: [B, F, T]
            pitch_pred: [B, T]
            energy_pred: [B, T]
            log_duration_pred: [B, T]
            duration_rounded_pred: [B, T]
            src_masks: [B, T]
            mel_masks: [B, T]
        """
        speakers, texts, src_lens, max_src_len, mels, mel_lens, max_mel_len = batch['speaker'], batch['text'], batch['text_length'], batch['text_max_length'], batch.get('mel'), batch.get('mel_length'), batch.get('mel_max_length')
        p_targets, e_targets, d_targets = batch.get('pitch'), batch.get('energy'), batch.get('duration')
        p_control, e_control, d_control = batch.get('p_control', 1.0), batch.get('e_control', 1.0), batch.get('d_control', 1.0)
        
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        if self.model_cfg.multi_speaker:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )
        
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        output = output.transpose(1, 2)
        postnet_output = postnet_output.transpose(1, 2)
        
        return dict(
            mel_pred = output,
            mel_postnet_pred = postnet_output,
            pitch_pred = p_predictions,
            energy_pred = e_predictions,
            log_duration_pred = log_d_predictions,
            duration_rounded_pred = d_rounded,
            text_mask = src_masks,
            mel_length = mel_lens,
            mel_mask = mel_masks,
        )
