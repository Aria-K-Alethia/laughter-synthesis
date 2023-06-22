'''
	Copyright (c) Xin Detai@University of Tokyo

	Description:
		TTS for laughter synthesis
	Licence:
		MIT
	THE USER OF THIS CODE AGREES TO ASSUME ALL LIABILITY FOR THE USE OF THIS CODE.
	Any use of this code should display all the info above.
'''
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import random
import hydra
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import join
from functools import partial
from itertools import chain
from collections import defaultdict
from model.fastspeech2 import FastSpeech2
from model.loss import FastSpeech2Loss
import utils


class BaselineLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ocwd = hydra.utils.get_original_cwd()
        self.lr = cfg.train.lr
        self.construct_model()
        self.criterion = FastSpeech2Loss(cfg)
        self.load_acoustic_stats()
        self.save_hyperparameters()

    def construct_model(self):
        self.model = FastSpeech2(self.cfg)
        self.vocoder = utils.get_vocoder_16k(self.cfg.model.vocoder.model, join(self.ocwd, self.cfg.model.vocoder.path))
        print(self.model)
    
    def load_acoustic_stats(self):
        path = join(self.ocwd, self.cfg.preprocess.path.processed_path, 'stats.pt')
        self.stats = torch.load(path)
    
    def forward(self, batch):
        output = self.model(batch)
        return output

    def on_train_epoch_start(self):
        pass

    def training_epoch_end(self, outputs):
        pass

    def training_step(self, batch, batch_idx):
        self.model.increment_step()
        output = self(batch)
        
        loss_dict = self.criterion(batch, output, self.model.step)
        loss = loss_dict['loss']
        
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfg.dataset.train.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        
        loss_dict = self.criterion(batch, output, self.model.step)

        # collect results and metrics
        out = {
            'mel_gt': batch['mel'].detach().cpu(),
            'mel_pred': output['mel_postnet_pred'].detach().cpu(),
            'pitch_gt': batch['pitch'].detach().cpu(),
            'pitch_pred': output['pitch_pred'].detach().cpu(),
            'energy_gt': batch['energy'].detach().cpu(),
            'energy_pred': output['energy_pred'].detach().cpu(),
            'mel_length': batch['mel_length'].detach().cpu(),
            'text_length': batch['text_length'].detach().cpu(),
            'duration': batch['duration'].detach().cpu(),
            'fid': batch['fid'],
            'raw_speaker': batch['raw_speaker'],
            'raw_text': batch['raw_text'],
            'raw_phone': batch['raw_phone'],
        }
        out.update({k: v.detach().cpu() if hasattr(v, 'detach') else v for k, v in loss_dict.items()})
        return out

    def get_mel_gst(self, spkrs, device):
        mels = [torch.from_numpy(torch.load(self.spkr2gst[spkr])).float() for spkr in spkrs]
        mel_lengths = torch.LongTensor([mel.shape[-1] for mel in mels]).to(device)
        mels = torch.stack(utils.pad_torch(mels)).to(device)
        return mels, mel_lengths
        
    def test_step(self, batch, batch_idx):
        # test step, pop pitch energy duration GT
        pitch, energy, duration = batch.pop('pitch'), batch.pop('energy'), batch.pop('duration')
        mel, mel_len, mel_max_len = batch.pop('mel'), batch.pop('mel_length'), batch.pop('mel_max_length')
        # change gst mel to test gst
        if self.cfg.model.use_gst:
            mel_gst, mel_gst_length = self.get_mel_gst(batch['raw_speaker'], batch['mel_gst'].device)
            batch['mel_gst'], batch['mel_gst_length'] = mel_gst, mel_gst_length
        # run the model
        output = self(batch)
        # register pitch, energy and duration
        batch['pitch'], batch['energy'], batch['duration'] = pitch, energy, duration
        batch['mel'], batch['mel_length'], batch['mel_max_length'] = mel, mel_len, mel_max_len
        
        #loss_dict = self.criterion(batch, output)
        
        # collect results and metrics
        out = {
            'wav': batch['wav'],
            'mel_gt': batch['mel'].detach().cpu(),
            'mel_pred': output['mel_postnet_pred'].detach().cpu(),
            'pitch_gt': batch['pitch'].detach().cpu(),
            'pitch_pred': output['pitch_pred'].detach().cpu(),
            'energy_gt': batch['energy'].detach().cpu(),
            'energy_pred': output['energy_pred'].detach().cpu(),
            'mel_length': output['mel_length'].detach().cpu(), # this length should get from model itself
            'mel_length_gt': batch['mel_length'].detach().cpu(),
            'text_length': batch['text_length'].detach().cpu(),
            'duration': batch['duration'].detach().cpu(),
            'duration_rounded_pred': output['duration_rounded_pred'].detach().cpu(),
            'fid': batch['fid'],
            'raw_speaker': batch['raw_speaker'],
            'raw_text': batch['raw_text'],
            'raw_phone': batch['raw_phone'],
        }
        #out.update({k: v.detach().cpu() for k, v in loss_dict.items()})
        
        return out

    def validation_epoch_end(self, outputs):
        # loss aggregate
        val_loss = torch.stack([item['loss'] for item in outputs]).mean().item()
        val_mel_loss = torch.stack([item['mel_loss'] for item in outputs]).mean().item()
        val_mel_postnet_loss = torch.stack([item['mel_postnet_loss'] for item in outputs]).mean().item()
        val_duration_loss = torch.stack([item['duration_loss'] for item in outputs]).mean().item()
        val_pitch_loss = torch.stack([item['pitch_loss'] for item in outputs]).mean().item()
        val_energy_loss = torch.stack([item['energy_loss'] for item in outputs]).mean().item()
        
        # wav reconstruction
        if self.current_epoch % self.cfg.train.audio_log_interval:
            i1 = random.randint(0, len(outputs)-1)
            batch = outputs[i1]
            i2 = random.randint(0, batch['mel_gt'].shape[0]-1)
            wav_recon, wav_pred, mel_gt_fig, mel_pred_fig = self.visualize(batch, i2)
            fid, speaker, raw_text = batch['fid'][i2], batch['raw_speaker'][i2], batch['raw_text'][i2]
        
        # log
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mel_loss', val_mel_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mel_postnet_loss', val_mel_postnet_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_pitch_loss', val_pitch_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_energy_loss', val_energy_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_duration_loss', val_duration_loss, on_epoch=True, prog_bar=True, logger=True)
        if self.current_epoch % self.cfg.train.audio_log_interval:
            self.log_figure(mel_gt_fig, 'val/mel_gt')
            self.log_figure(mel_pred_fig, 'val/mel_pred')
            self.log_audio(wav_recon, 'val/wav_gt', caption=f'{fid}')
            self.log_audio(wav_pred, 'val/wav_pred', caption=f'{fid}')
        
    def test_epoch_end(self, outputs):
        # synthesize all
        speaker2wavs = defaultdict(list)
        for batch in outputs:
            mel_pred, mel_length = batch['mel_pred'], batch['mel_length']
            raw_speakers, fids, texts = batch['raw_speaker'], batch['fid'], batch['raw_text']
            gt_wavs = batch['wav']
            wavs = self.synthesize(mel_pred.to(self.device).float(), mel_length)
            #gt_wavs = self.synthesize(batch['mel_gt'].to(self.device).float(), batch['mel_length_gt'])
            for wav, gt_wav, spkr, fid, text in zip(wavs, gt_wavs, raw_speakers, fids, texts):
                gt_wav = (gt_wav * self.cfg.preprocess.audio.max_wav_value).astype(np.int16)
                pack = {'wav': wav, 'fid': fid, 'gt_wav': gt_wav, 'text': text}
                speaker2wavs[spkr].append(pack)
                
        # visualize random mel
        n = 5
        for batch in random.sample(outputs, n):
            mel_pred, mel_length, pitch, energy = batch['mel_pred'], batch['mel_length'], batch['pitch_pred'], batch['energy_pred']
            mel_gt, mel_length_gt, pitch_gt, energy_gt = batch['mel_gt'], batch['mel_length_gt'], batch['pitch_gt'], batch['energy_gt']
            i = random.randint(0, mel_pred.shape[0]-1)
            l1, l2 = mel_length[i].item(), mel_length_gt[i].item()
            d1 = batch['duration_rounded_pred'][i, :l1].numpy()
            d2 = batch['duration'][i, :l2].numpy()
            spkr = batch['raw_speaker'][i]
            fig1 = self.plot_mel(mel_pred[i, :, :l1].numpy(), utils.expand(pitch[i, :l1], d1) , utils.expand(energy[i, :l1], d1), spkr, 'pred_mel')
            fig2 = self.plot_mel(mel_gt[i, :, :l2].numpy(), utils.expand(pitch_gt[i, :l2], d2), utils.expand(energy_gt[i, :l2], d2), spkr, 'gt_mel')
            self.log_figure(fig1, 'test/mel_pred')
            self.log_figure(fig2, 'test/mel_gt')
            
        for spkr, packs in speaker2wavs.items():
            i = random.randint(0, len(packs) - 1)
            pack = packs[i]
            pred, gt = pack['wav'], pack['gt_wav']
            fid = pack['fid']
            self.log_audio(pred, 'test/wav_pred', caption=f'{fid}')
            self.log_audio(gt, 'test/wav_gt', caption=f'{fid}')
        

    def configure_optimizers(self):
        optimizer = optim.Adam([
                    {'params':self.model.parameters(), 'lr': self.cfg.train.lr},])
        scheduler = utils.get_transformer_scheduler(optimizer, self.cfg.train.warmup_steps, self.cfg.model.transformer.encoder_hidden, verbose=False)
        scheduler_config = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}

    def visualize(self, batch, i):
        gt_mels = batch['mel_gt']
        mel_length, text_length = batch['mel_length'][i].item(), batch['text_length'][i].item()
        gt_mel, pred_mel  = gt_mels[i, :, :mel_length], batch['mel_pred'][i, :, :mel_length]
        duration = batch['duration'][i, :text_length].numpy()
        if self.cfg.preprocess.pitch.feature == 'phoneme':
            pitch_gt, pitch_pred = batch['pitch_gt'][i, :text_length].numpy(), batch['pitch_pred'][i, :text_length].numpy()
            pitch_gt, pitch_pred = utils.expand(pitch_gt, duration), utils.expand(pitch_pred, duration)
        else:
            pitch_gt, pitch_pred = batch['pitch_gt'][i, :mel_length].numpy(), batch['pitch_pred'][i, :mel_length].numpy()
            
        if self.cfg.preprocess.energy.feature == 'phoneme':
            energy_gt, energy_pred = batch['energy_gt'][i, :text_length].numpy(), batch['energy_pred'][i, :text_length].numpy()
            energy_gt, energy_pred = utils.expand(energy_gt, duration), utils.expand(energy_pred, duration)
        else:
            energy_gt, energy_pred = batch['energy_gt'][i, :mel_length].numpy(), batch['energy_pred'][i, :mel_length].numpy()
        # generate wav
        wav_recon = self.synthesize(gt_mel.to(self.device).float())[0]
        wav_pred = self.synthesize(pred_mel.to(self.device).float())[0]
    
        # plot mel
        spkr = batch['raw_speaker'][i]
        mel_gt_fig = self.plot_mel(gt_mel.numpy(), pitch_gt, energy_gt, spkr, 'gt_mel')
        mel_pred_fig = self.plot_mel(pred_mel.numpy(), pitch_pred, energy_pred, spkr, 'pred_mel')
        
        return wav_recon, wav_pred, mel_gt_fig, mel_pred_fig
    
    def synthesize(self, mel, lengths=None):
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        self.vocoder.eval()
        wavs = self.vocoder(mel).squeeze(1)
        wavs = wavs.detach().cpu().numpy() * self.cfg.preprocess.audio.max_wav_value
        wavs = wavs.astype('int16')
        wavs = [wav for wav in wavs]
        
        if lengths is not None:
            for i in range(mel.shape[0]):
                wavs[i] = wavs[i][:lengths[i] * self.cfg.preprocess.stft.hop_length]

        return wavs
        
    def recover_pitch(self, pitch, mean, std):
        method = self.cfg.preprocess.pitch.norm_method
        if method == 'z_score':
            pitch = pitch * std + mean
        elif method == 'mean':
            pitch = pitch + mean
        elif method == 'none':
            pass
        else:
            raise ValueError(f'No such pitch norm method: {method}')
        return pitch
    
    def plot_mel(self, mel, pitch, energy, spkr, title=None):
        """plot mel-spec with pitch and energy curve

        Args:
            mel (np.array): mel-spec with [#n_mel, T]
            pitch (np.array): pitch [T]
            energy (np.array): energy [T]
            title (string, optional): title. Defaults to None.
        Returns:
            matplotlib figure obj
        """
        fig, axes = plt.subplots(1, 1, squeeze=False)
        axis = axes[0][0]
        pmin, pmax, pmean, pstd, emin, emax = self.stats['pitch']['min'], self.stats['pitch']['max'], self.stats['pitch']['mean'][spkr], self.stats['pitch']['std'][spkr], self.stats['energy']['min'], self.stats['energy']['max']
        
        pmin = self.recover_pitch(pmin, pmean, pstd)
        pmax = self.recover_pitch(pmax, pmean, pstd)
        pitch = self.recover_pitch(pitch, pmean, pstd)

        def add_axis(fig, old_axis):
            axis = fig.add_axes(old_axis.get_position(), anchor='W')
            axis.set_facecolor('None')
            return axis
        
        axis.imshow(mel, origin='lower')
        axis.set_aspect(2.5, adjustable='box')
        axis.set_ylim(0, mel.shape[0])
        axis.set_title(title, fontsize='medium')
        axis.tick_params(labelsize='x-small', left=False, labelleft=False)
        
        ax1 = add_axis(fig, axis)
        ax1.plot(pitch, color='tomato')
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pmax)
        ax1.set_ylabel('F0', color='tomato')
        ax1.tick_params(labelsize='x-small', colors='tomato', bottom=False, labelbottom=False)
        
        ax2 = add_axis(fig, axis)
        ax2.plot(energy, color='darkviolet')
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(emin, emax)
        ax2.set_ylabel('Energy', color='darkviolet')
        ax2.yaxis.set_label_position('right')
        ax2.tick_params(labelsize='x-small', colors='darkviolet', bottom=False, labelbottom=False, left=False, labelleft=False, right=True, labelright=True)
        
        return fig

    def log_figure(self, fig, tag):
        if self.cfg.use_tb:
            self.loggers[0].experiment.add_figure(tag, fig, self.current_epoch)
    
    def log_audio(self, audio, tag, caption=None):
        sr = self.cfg.preprocess.audio.sr
        audio = audio / self.cfg.preprocess.audio.max_wav_value
        if self.cfg.use_tb:
            self.loggers[0].experiment.add_audio(tag, audio, self.current_epoch, sr)
