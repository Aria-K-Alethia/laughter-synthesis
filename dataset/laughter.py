import os
import librosa
from glob import glob
from os.path import join, basename, exists, expanduser
from collections import defaultdict

class Laughter:
    _dirs = ['denoised']
    
    def __init__(self, root):
        self.root = expanduser(root)
        self.filelist = self.collect_files()

    def collect_files(self):
        files = []
        for d in self._dirs:
            path = join(self.root, d, '*.wav')
            fs = glob(path)
            for f in fs:
                dur = librosa.get_duration(filename=f)
                if dur != 0:
                    files.append(f)
        return files
    
    def get_speaker2fids(self):
        spkr2wavs = defaultdict(list)
        for fid in self.filelist:
            fid = basename(fid)
            spkr = self.get_speaker(fid)
            spkr2wavs[spkr].append(fid[:-4])
        return spkr2wavs
        
    @staticmethod
    def get_speaker(fid):
        spkr = '-'.join(fid.split('-')[:-1])
        return spkr

    def collect_speakers(self):
        s = set()
        for f in self.filelist:
            f = basename(f)
            spkr = self.get_speaker(f)
            s.add(spkr)
        s = list(s)
        s.sort()
        return s

if __name__ == '__main__':
    root = '~/Downloads/laughter_open'
    ds = Laughter(root)
    print(len(ds.collect_speakers()))
    spkr2fids = ds.get_speaker2fids()
    spkr2fids = sorted(list({k: len(v) for k, v in spkr2fids.items()}.items()), key=lambda x: x[1])
    print(spkr2fids)
