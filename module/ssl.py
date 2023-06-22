import torch
import torch.nn as nn
from utils import load_ssl_model

class SSLWrapper(nn.Module):
    '''
        Wrapper for wav2vec2 and HuBERT
        
        Args:
            model_path: hugging face model path
            layer: extract
    '''
    def __init__(self, model_path):
        super().__init__()
        self.ssl = load_ssl_model(model_path)
        self.config = self.ssl.config
        self.num_hidden_layers = self.config.num_hidden_layers

    def forward(self, wav):
        # wav: [B, T]
        out = self.ssl(wav, output_hidden_states=True)
        #feat = out.hidden_states[-1]
        feat = torch.stack(out.hidden_states[-self.num_hidden_layers:]) # [layer, B, L, F]
        return feat
