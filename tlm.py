import os
import torch
import utils
import sys
from os.path import exists, join, basename

trainfile = './filelists/laughter_train.txt'
valfile = './filelists/laughter_val.txt'
testfile = './filelists/laughter_test.txt'

name2list = {'train': trainfile, 'val': valfile, 'test': testfile}
name2list = {k: utils.read_filelist(f) for k, f in name2list.items()}
name2code = {}
max_length = float('-inf')
for k, filelist in name2list.items():
    buf = []
    for (f,) in filelist:
        with open(join(f'./data/laughter', 'code', f+'.txt')) as file:
            code = file.read().strip()
            max_length = max(max_length, len(code.split()))
            buf.append((code,))
    print(f'{k}, {len(buf)} items')
    name2code[k] = buf
print(f'Max sample length: {max_length}')
outdir = f'tlm/data'
os.makedirs(outdir, exist_ok=True)
for name, code in name2code.items():
    outpath = join(outdir, name+'.txt')
    utils.write_filelist(code, outpath)
