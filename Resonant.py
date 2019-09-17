#!/usr/bin/env python
# coding: utf-8

from batch import *

import awkward as awk
import numpy as np
import uproot
from nanoframe import NanoFrame
import os
import particle
import pandas as pd
import uproot_methods


final_dfs = {
    'pf' : pd.DataFrame(),
    'lpt' : pd.DataFrame(),
}

branches = ['BToKEE_charge',
 'BToKEE_chi2',
 'BToKEE_fit_cos2D',
 'BToKEE_fit_eta',
 'BToKEE_fit_mass',
 'BToKEE_fit_phi',
 'BToKEE_fit_pt',
 'BToKEE_kIdx',
 'BToKEE_l1Idx',
 'BToKEE_l2Idx',
 'BToKEE_l_xy',
 'BToKEE_l_xy_unc',
 'BToKEE_mll_fullfit',
 'BToKEE_mll_llfit',
 'BToKEE_mll_raw',
 'BToKEE_mass',
 'BToKEE_eta',
 'BToKEE_phi',
 'BToKEE_pt',
 'BToKEE_svprob',
 'Electron_isLowPt',
 'Electron_isPF',
 'Electron_eta',
 'Electron_mass',
 'Electron_mvaId',
 'Electron_phi',
 'Electron_pt',
 'Electron_vz',
 'Muon_pt',
 'Muon_mass',
 'Muon_eta',
 'Muon_phi',
 'Muon_pt',
 'Muon_vz',
 'Muon_isTriggering',
 'ProbeTracks_DCASig',
 'ProbeTracks_eta',
 'ProbeTracks_mass',
 'ProbeTracks_phi',
 'ProbeTracks_pt',
 'ProbeTracks_vz',
 'nBToKEE',
 'nElectron',
 'nMuon',
 'nProbeTracks',
 'HLT_*',
]

nprocessed = 0

from pdb import set_trace
for fname in infiles:
    print('processing:', fname)
    nf = NanoFrame(fname, branches = branches)
    
    # Load the needed collections, NanoFrame is just an empty shell until we call the collections
    
    muons = nf['Muon']
    electrons = nf['Electron']
    tracks = nf['ProbeTracks']
    hlt = nf['HLT']
    bcands = nf['BToKEE']
    
    nprocessed += hlt.shape[0]
    # Attach the objects to the candidates
    bcands['e1'] = electrons[bcands['l1Idx']]
    bcands['e2'] = electrons[bcands['l2Idx']]
    bcands['k'] = tracks[bcands['kIdx']]
    bcands['p4fit'] = uproot_methods.TLorentzVectorArray.from_ptetaphim(
        bcands['fit_pt'], bcands['fit_eta'], bcands['fit_phi'], bcands['fit_mass']
    )
    
    
    # Attach the trigger muon, identified as the closest in dz to the lead electron
    muon_trg_mask = (muons.isTriggering == 1)
    for path, pt_thr in [('Mu8_IP5', 8), ('Mu10p5_IP3p5', 10), ('Mu8_IP3', 8), ('Mu8p5_IP3p5', 8.5), 
                         ('Mu9_IP5', 9), ('Mu7_IP4', 7), ('Mu9_IP4', 9), ('Mu9_IP6', 9), 
                         ('Mu8_IP6', 8), ('Mu12_IP6', 12)]:
        if not any(path in i for i in hlt.columns): # the trigger is not here
            continue
        else:
            #merge all the parts and compute an or
            hlt_fired = np.hstack(
                [hlt[i].reshape((hlt[i].shape[0], 1)) for i in hlt.columns if path in i]
            ).any(axis = 1)
            muon_trg_mask = muon_trg_mask | (hlt_fired & (muons.p4.pt > pt_thr))
    
    one_trg_muon = (muon_trg_mask.sum() != 0)
    trig_mu = muons[muon_trg_mask][one_trg_muon]
    bcands = bcands[one_trg_muon]
    
    e1z, muz = bcands.e1.vz.cross(trig_mu.vz, nested = True).unzip()
    closest_mu = np.abs(e1z - muz).argmin().flatten(axis = 1)
    bcands['trg_mu'] = trig_mu[closest_mu]
    
    
    # Candidate selection, cut-based for the moment
    
    b_selection = (bcands.k.p4.pt > 1.5) & (bcands.p4fit.pt > 3) & \
                  (bcands.svprob > 0.1) & (bcands.fit_cos2D > 0.999) & \
                  ((bcands.l_xy / bcands.l_xy_unc) > 6)
    
    b_pf = bcands.e1.isPF & bcands.e2.isPF
    b_lpt = bcands.e1.isLowPt & bcands.e2.isLowPt & (bcands.e1.mvaId > 3.96) & (bcands.e2.mvaId > 3.96)
    
    best_pf_cand = bcands[b_selection & b_pf].svprob.argmax()
    bcands_pf = (bcands[b_selection & b_pf][best_pf_cand]).flatten()
    
    best_lpt_cand = bcands[b_selection & b_lpt].svprob.argmax()
    bcands_lpt = (bcands[b_selection & b_lpt][best_lpt_cand]).flatten()
    
    dfs = {
        'pf' : pd.DataFrame(),
        'lpt' : pd.DataFrame(),
    }
    
    for name, tab, sel in [('pf', bcands_pf, b_selection & b_pf), ('lpt', bcands_lpt, b_selection & b_lpt)]:
        df = dfs[name]
        df['e1pt'] = tab.e1.p4.pt
        df['e2pt'] = tab.e2.p4.pt
        df['e1pf'] = tab.e1.isPF
        df['e2pf'] = tab.e2.isPF
        df['e1mvaId'] = tab.e1.mvaId
        df['e2mvaID'] = tab.e2.mvaId
        df['kpt'] = tab.k.p4.pt
        df['kDCA'] = tab.k.DCASig
        df['Bcharge'] = tab.charge
        df['Bpt'] = tab.p4fit.pt
        df['Beta'] = tab.p4fit.eta
        df['Bsvprob'] = tab.e2.p4.pt
        df['Bcos2D'] = tab.fit_cos2D
        df['Blxy_sig'] = (tab.l_xy / tab.l_xy_unc)
        df['Bmll'] = tab.mll_fullfit
        df['Bmll_raw'] = tab.mll_raw
        df['Bmll_llfit'] = tab.mll_llfit
        df['Bmass'] = tab.fit_mass
        df['Bmass_raw'] = tab.p4.mass
        df['trgmu_eta'] = tab.trg_mu.p4.eta
        df['trgmu_pt'] = tab.trg_mu.p4.pt
        df['nB'] = sel.sum()[sel.sum() != 0]
    
    final_dfs['lpt'] = pd.concat((final_dfs['lpt'], dfs['lpt']))
    final_dfs['pf'] = pd.concat((final_dfs['pf'], dfs['pf']))

final_dfs['lpt'].to_hdf(args.f_out, 'lpt', mode = 'a')
final_dfs['pf'].to_hdf(args.f_out, 'pf', mode = 'a')
print('DONE! Processed events: ', nprocessed)
print('PF size:', final_dfs['pf'].shape[0], 'LPT size:', final_dfs['lpt'].shape[0])
