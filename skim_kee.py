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
 'BToKEE_fit_mass',
 'BToKEE_fit_massErr',
 'BToKEE_fit_pt',
 'BToKEE_fit_eta',
 'BToKEE_fit_phi',
 'BToKEE_kIdx',
 'BToKEE_l1Idx',
 'BToKEE_l2Idx',
 'BToKEE_l_xy',
 'BToKEE_l_xy_unc',
 'BToKEE_mll_fullfit',
 'BToKEE_mll_llfit',
 'BToKEE_mll_raw',
 'BToKEE_mllErr_llfit',
 'BToKEE_mass',
 'BToKEE_eta',
 'BToKEE_phi',
 'BToKEE_pt',
 'BToKEE_svprob',
 'BToKEE_fit_k_eta',
 'BToKEE_fit_k_pt', 
 'BToKEE_fit_k_phi', 
 'BToKEE_fit_l1_eta',
 'BToKEE_fit_l1_pt',
 'BToKEE_fit_l1_phi', 
 'BToKEE_fit_l2_eta',
 'BToKEE_fit_l2_pt',
 'BToKEE_fit_l2_phi',

 'Electron_isLowPt',
 'Electron_isPF',
 'Electron_isPFoverlap',
 'Electron_mass',
 'Electron_eta',
 'Electron_phi',
 'Electron_pt',
 'Electron_dxy',
 'Electron_dxyErr',
 'Electron_mvaId',
 'Electron_ptBiased',
 'Electron_unBiased',
 'Muon_pt',
 'Muon_eta',
 'Muon_phi',
 'Muon_mass',
 'Muon_pt',
 'Muon_vz',
 'Muon_isTriggering',
 'ProbeTracks_DCASig',
 'ProbeTracks_dxyS',
 'ProbeTracks_pt',
 'ProbeTracks_eta',
 'ProbeTracks_phi',
 'ProbeTracks_mass',
 'ProbeTracks_vz',
 'ProbeTracks_isMatchedToEle',
 'ProbeTracks_isMatchedToSoftMuon',
 'nBToKEE',
 'nElectron',
 'nMuon',
 'nProbeTracks',
 'HLT_*',
]

if args.mc:
    branches.extend([
        'nGenPart',    ## always load n for the vector length
        'GenPart_pdgId',
        'GenPart_pt',
        'GenPart_eta',
        'GenPart_phi',
        'GenPart_mass',
        'GenPart_genPartIdxMother',
        'ProbeTracks_genPartIdx',
        'Muon_genPartIdx',
        'Electron_genPartIdx',
    ])
nprocessed = 0

from pdb import set_trace
for fname in infiles:
    print('processing:', fname)
    nf = NanoFrame(fname, branches = branches)
    # Load the needed collections, NanoFrame is just an empty shell until we call the collections
    
    muons     = nf['Muon']
    electrons = nf['Electron']
    tracks    = nf['ProbeTracks']
    hlt       = nf['HLT']
    bcands    = nf['BToKEE']

    nprocessed += hlt.shape[0]
    gen       = nf['GenPart']

    # MC Matching
    if args.mc:
        # electrons
        ele_gen_idx = (electrons.genPartIdx != -1) * electrons.genPartIdx
        electrons['genPdgId'] = gen[ ele_gen_idx ].pdgId
        electrons['isGenEle'] = abs(electrons['genPdgId']) == 11
        
        mom_gen_idx = (gen[ele_gen_idx].genPartIdxMother != -1) * (gen[ele_gen_idx].genPartIdxMother)
        #     hasMother   = gen[ ele_gen_idx ].genPartIdxMother != -1
        electrons['motherPdgId'] = gen[mom_gen_idx].pdgId
        
        
        jpsi_ele = electrons['isGenEle'] * (electrons.motherPdgId == 443)
        gma_gen_idx = (gen[mom_gen_idx].genPartIdxMother != -1) * (gen[mom_gen_idx].genPartIdxMother)
        jpsi_mum = jpsi_ele * gen[gma_gen_idx] 
        electrons['granmaPdgId'] = jpsi_mum.pdgId
        
        trk_gen_idx = (tracks.genPartIdx != -1) * tracks.genPartIdx
        tracks['genPdgId']       = gen[trk_gen_idx].pdgId
        trk_mom_gen_idx = (gen[trk_gen_idx].genPartIdxMother != -1) * (gen[trk_gen_idx].genPartIdxMother)
        tracks['motherPdgId']    = gen[trk_mom_gen_idx].pdgId

    # Attach the objects to the candidates
    bcands['e1'] = electrons[bcands['l1Idx']]
    bcands['e2'] = electrons[bcands['l2Idx']]
    bcands['k'] = tracks[bcands['kIdx']]
    bcands['p4fit'] = uproot_methods.TLorentzVectorArray.from_ptetaphim(
        bcands['fit_pt'], bcands['fit_eta'], bcands['fit_phi'], bcands['fit_mass']
    )
    bcands['event']           = nf['event']
    bcands['run']             = nf['run']
    bcands['luminosityBlock'] = nf['luminosityBlock']    

    l_xy_sig = bcands.l_xy / bcands.l_xy_unc
    l_xy_sig[np.invert(np.isfinite(l_xy_sig))] = -99
    bcands['l_xy_sig'] = l_xy_sig

    
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
    b_selection = (bcands.e1.mvaId > -10.) & (bcands.e2.mvaId > -10.)  &\
                  (bcands.fit_mass > 4.8) #&\
    ##              (bcands.k.isMatchedToEle == 0) & (bcands.k.isMatchedToSoftMuon == 0) & \
    ##              (bcands.l_xy_sig > 1.)
    ##              (bcands.fit_k_pt > 3) & (bcands.k.DCASig > 2) & (bcands.fit_pt > 3) & \
    ##              (bcands.svprob > 0.1) & (bcands.fit_cos2D > 0.999) & \
    
    ## For data: keep only one in 1000 events
    if not args.mc:
        b_selection = b_selection & (nf['event'] % 1000 == 0)

    sel_bcands = bcands[b_selection].flatten()

    ## we should find a way to remove the candidates with lowPt eles overlapped with PF only
    ## if the corresponding B candidate made from PFs is passing pre-selections
    
    df = pd.DataFrame()
    df['event'] = sel_bcands['event']
    df['run'] = sel_bcands['run']
    df['luminosityBlock'] = sel_bcands['luminosityBlock']
    
    df['e1_pt'       ] = sel_bcands.fit_l1_pt
    df['e2_pt'       ] = sel_bcands.fit_l2_pt
    df['k_pt'        ] = sel_bcands.fit_k_pt
    df['e1_phi'      ] = sel_bcands.fit_l1_phi
    df['e2_phi'      ] = sel_bcands.fit_l2_phi
    df['k_phi'       ] = sel_bcands.fit_k_phi
    df['e1_eta'      ] = sel_bcands.fit_l1_eta
    df['e2_eta'      ] = sel_bcands.fit_l2_eta
    df['k_eta'       ] = sel_bcands.fit_k_eta
    
    df['e1_isPF'     ] = sel_bcands.e1.isPF
    df['e2_isPF'     ] = sel_bcands.e2.isPF
    df['e1_pfOverlap'] = sel_bcands.e1.isPFoverlap
    df['e2_pfOverlap'] = sel_bcands.e2.isPFoverlap
    df['e1_mvaId'    ] = sel_bcands.e1.mvaId
    df['e2_mvaId'    ] = sel_bcands.e2.mvaId
    df['e1_unBDT'    ] = sel_bcands.e1.unBiased
    df['e2_unBDT'    ] = sel_bcands.e2.unBiased

    e1_dxyS = sel_bcands.e1.dxy/sel_bcands.e1.dxyErr
    e1_dxyS[np.invert(np.isfinite(e1_dxyS))] = -99
    e2_dxyS = sel_bcands.e2.dxy/sel_bcands.e2.dxyErr
    e2_dxyS[np.invert(np.isfinite(e2_dxyS))] = -99
    df['e1_dxyS'    ] =  e1_dxyS
    df['e2_dxyS'    ] =  e2_dxyS

        
    df['k_DCA'       ] = sel_bcands.k.DCASig
    df['k_dxyS'      ] = sel_bcands.k.dxyS

    df['B_charge'    ] = sel_bcands.charge
    df['B_mass'      ] = sel_bcands.fit_mass
    df['B_mass_err'  ] = sel_bcands.fit_massErr
    df['B_pt'        ] = sel_bcands.fit_pt
    df['B_eta'       ] = sel_bcands.fit_eta
    df['B_phi'       ] = sel_bcands.fit_phi
    df['B_svprob'    ] = sel_bcands.svprob
    df['B_cos'       ] = sel_bcands.fit_cos2D
    df['B_ls'        ] = sel_bcands.l_xy_sig
    df['B_mll'       ] = sel_bcands.mll_llfit

    if args.mc:
        df['k_genPdgId'  ] = sel_bcands.k.genPdgId 
        df['e1_genPdgId' ] = sel_bcands.e1.genPdgId 
        df['e2_genPdgId' ] = sel_bcands.e2.genPdgId
        df['k_genMumId'  ] = sel_bcands.k.motherPdgId
        df['e1_genMumId' ] = sel_bcands.e1.motherPdgId
        df['e2_genMumId' ] = sel_bcands.e2.motherPdgId
        df['e1_genGMaId' ] = sel_bcands.e1.granmaPdgId
        df['e2_genGMaId' ] = sel_bcands.e2.granmaPdgId
    

    # df['trgmu_eta'] = sel_bcands.trg_mu.p4.eta
    # df['trgmu_pt'] = sel_bcands.trg_mu.p4.pt
    
    final_df = pd.concat((final_df, df))

# final_df.to_hdf(args.f_out, 'df', mode = 'w')
print('DONE! Processed events: ', nprocessed)
print('Saved events:', final_df.shape[0])

# import pdb; pdb.set_trace()
import numpy as np
# convert all unsigned integer to signed, as the streaming is not implemented yet
unsigned_patch = {np.dtype(f'uint{i}') : np.dtype(f'int{i}') for i in [8, 16, 32, 64]}
out = uproot.recreate(f_out)#, compression = uproot.LZMA(8))
out['tree'] = uproot.newtree({
    c : unsigned_patch.get(final_df[c].dtype, final_df[c].dtype) 
    for c in final_df.columns
})
out['tree'].extend({c : final_df[c].values for c in final_df.columns})

# # out['tree'] = final_final_df
# # uproot.newtree({'a' : np.int32, 'b' : np.float32})
# # out["tree"].extend({'a' : np.array([1,2,3,4]), 'b' : np.array([1.1, 2.2, 3.3, 4.4])})
out.close()

