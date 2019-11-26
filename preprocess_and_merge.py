#! /bin/env python

from argparse import ArgumentParser
import math

parser = ArgumentParser()
parser.add_argument('f_out', help='file path', default = 'test.root')
parser.add_argument('f_in', nargs = '+', help='file path')
parser.add_argument('--mc', action = 'store_true')
parser.add_argument('--resonant', action = 'store_true')
args = parser.parse_args()

import numpy as np
import uproot

branches = [
    'e1_mvaId',
    'e2_mvaId',
    'e1_unBDT',
    'e2_unBDT',
    'e1_dxyS',
    'e2_dxyS',
    'e1_isPF',
    'e2_isPF',
    'k_DCA', 
    'B_cos', 
    'B_ls', 
    'B_minDR', 
    'B_maxDR', 
    'B_svprob',
    'e1_isPF',
    'e2_isPF',
    'e1_pt',
    'e2_pt',
    'k_pt',
    'e1_eta',
    'e2_eta',
    'k_eta',
    'event',
    'B_mll',
    'B_mass',
    'B_pt',
    'B_eta',
    'e1_pfOverlap',
    'e2_pfOverlap',
]

mc_feats = [
    'e1_genPdgId',
    'e2_genPdgId',
    'e1_genMumId',
    'e2_genMumId',
    'k_genPdgId',
    'e1_genGMaId',
    'e2_genGMaId',
    'k_genMumId',
]

# hardcoded values!!
B_mc_mass = 5.26
B_mc_sigma = 0.06

unsigned_patch = {np.dtype(f'uint{i}') : np.dtype(f'int{i}') for i in [8, 16, 32, 64]}

out = uproot.recreate(args.f_out)#, compression = uproot.LZMA(8))
first = True

for rfile in args.f_in:
    tf = uproot.open(rfile)
    tt = tf['tree']
    arrays = tt.arrays(branches + mc_feats if args.mc else branches)
    raw = {i.decode() : j for i, j in arrays.items()}
    
    # create tree for the first file only
    if first:
        first = False
        out['tree'] = uproot.newtree({
            c : unsigned_patch.get(a.dtype, a.dtype) 
            for c, a in raw.items()
        })
        
    
    # select only matched candidates
    if args.mc:
        if args.resonant:
            # require all three candidates to come from the same B
            same_mother = (raw['e1_genGMaId'] == raw['e2_genGMaId']) & \
                          (raw['e1_genGMaId'] == raw['k_genMumId']) & (np.abs(raw['e1_genGMaId']) == 521)
        else:
            # require all three candidates to come from the same B and have low q**2
            same_mother = (raw['e1_genMumId'] == raw['e1_genMumId']) & \
                          (raw['e1_genMumId'] == raw['k_genMumId']) & (np.abs(raw['e1_genMumId']) == 521) & \
                          (raw['B_mll'] < 2.45)

        # Check all MC matches and B mass within 3 sigmas
        mask = ( (np.abs(raw['e1_genPdgId']) == 11) & (np.abs(raw['e2_genPdgId']) == 11) & 
                 (raw['e1_genMumId'] == 443) & (raw['e2_genMumId'] == 443) & 
                 (np.abs(raw['k_genPdgId']) == 321) & same_mother &
                 (np.abs(raw['B_mass'] - B_mc_mass) < 3 * B_mc_sigma)
             )
    else:
        # Require B Mass sidebands for the data
        mass_diff = np.abs(raw['B_mass'] - B_mc_mass) / B_mc_sigma
        mask = (3 < mass_diff) & (mass_diff < 7) & (raw['B_mll'] < 2.45)
        
    out['tree'].extend({
        c : a[mask]
        for c, a in raw.items()
    })
    # print(rfile, ':', raw['event'].shape[0], '-->', mask.sum())

out.close()

