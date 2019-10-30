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


final_df = pd.DataFrame()

branches = [
 'event',
 'run',
 'luminosityBlock', 
 'BToKEE_charge',
 'BToKEE_chi2',
 'BToKEE_fit_cos2D',
 'BToKEE_cos2D',
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
 'Electron_isPFoverlap',
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
    bcands['event'] = nf['event']
    bcands['run'] = nf['run']
    bcands['luminosityBlock'] = nf['luminosityBlock']    
    bcands['l_xy_sig'] = bcands.l_xy / np.sqrt(bcands.l_xy_unc)

    # Attach the trigger muon, identified as the closest in dz to the lead electron
    ## muon_trg_mask = (muons.isTriggering == 1)
    ## for path, pt_thr in [('Mu8_IP5', 8), ('Mu10p5_IP3p5', 10), ('Mu8_IP3', 8), ('Mu8p5_IP3p5', 8.5), 
    ##                      ('Mu9_IP5', 9), ('Mu7_IP4', 7), ('Mu9_IP4', 9), ('Mu9_IP6', 9), 
    ##                      ('Mu8_IP6', 8), ('Mu12_IP6', 12)]:
    ##     if not any(path in i for i in hlt.columns): # the trigger is not here
    ##         continue
    ##     else:
    ##         #merge all the parts and compute an or
    ##         hlt_fired = np.hstack(
    ##             [hlt[i].reshape((hlt[i].shape[0], 1)) for i in hlt.columns if path in i]
    ##         ).any(axis = 1)
    ##         muon_trg_mask = muon_trg_mask | (hlt_fired & (muons.p4.pt > pt_thr))
    ## 
    ## one_trg_muon = (muon_trg_mask.sum() != 0)
    ## trig_mu = muons[muon_trg_mask][one_trg_muon]
    ## bcands = bcands[one_trg_muon]
    ##
    ## e1z, muz = bcands.e1.vz.cross(trig_mu.vz, nested = True).unzip()
    ## closest_mu = np.abs(e1z - muz).argmin().flatten(axis = 1)
    ## bcands['trg_mu'] = trig_mu[closest_mu]
    
    
    # Candidate selection, cut-based for the moment
    
    b_selection = np.invert(bcands.e1.isPFoverlap) & np.invert(bcands.e2.isPFoverlap) & \
                  (bcands.e1.mvaId > 3.96) & (bcands.e2.mvaId > 3.96) & \
                  (bcands.k.p4.pt > 3) & (bcands.k.DCASig > 2) & (bcands.p4.pt > 3) & \
                  (bcands.svprob > 0.1) & (bcands.cos2D > 0.999) & \
                  (bcands.l_xy_sig > 6)
        
    best_cand = bcands[b_selection].svprob.argmax()
    # bcands_pf = (bcands[b_selection & b_pf][best_pf_cand]).flatten()
    sel_bcands = (bcands[b_selection][best_cand]).flatten()
    
    nBs = b_selection.sum()
    df = pd.DataFrame()
    df['event'] = sel_bcands['event']
    df['run'] = sel_bcands['run']
    df['luminosityBlock'] = sel_bcands['luminosityBlock']
    df['e1pt'] = sel_bcands.e1.p4.pt
    df['e2pt'] = sel_bcands.e2.p4.pt
    df['e1pf'] = sel_bcands.e1.isPF
    df['e2pf'] = sel_bcands.e2.isPF
    df['e1pfOverlap'] = sel_bcands.e1.isPFoverlap
    df['e2pfOverlap'] = sel_bcands.e2.isPFoverlap
    df['e1mvaId'] = sel_bcands.e1.mvaId
    df['e2mvaID'] = sel_bcands.e2.mvaId
    df['kpt'] = sel_bcands.k.p4.pt
    df['kDCA'] = sel_bcands.k.DCASig
    df['Bcharge'] = sel_bcands.charge
    df['Bpt'] = sel_bcands.p4.pt
    df['Beta'] = sel_bcands.p4.eta
    df['Bsvprob'] = sel_bcands.svprob
    df['Bcos2D'] = sel_bcands.cos2D
    df['Blxy_sig'] = sel_bcands.l_xy_sig
    df['Bmll'] = sel_bcands.mll_raw
    df['Bmass'] = sel_bcands.p4.mass
    df['nB'] = nBs[nBs != 0]
    # df['trgmu_eta'] = sel_bcands.trg_mu.p4.eta
    # df['trgmu_pt'] = sel_bcands.trg_mu.p4.pt
    
    final_df = pd.concat((final_df, df))

final_df.to_hdf(args.f_out, 'df', mode = 'w')
print('DONE! Processed events: ', nprocessed)
print('Saved events:', final_df.shape[0])
