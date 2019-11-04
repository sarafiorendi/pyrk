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
 'Electron_genPartIdx',
 'Muon_pt',
 'Muon_eta',
 'Muon_phi',
 'Muon_mass',
 'Muon_pt',
 'Muon_vz',
 'Muon_isTriggering',
 'Muon_genPartIdx',
 'ProbeTracks_DCASig',
 'ProbeTracks_dxyS',
 'ProbeTracks_pt',
 'ProbeTracks_eta',
 'ProbeTracks_phi',
 'ProbeTracks_mass',
 'ProbeTracks_vz',
 'ProbeTracks_isMatchedToEle',
 'ProbeTracks_isMatchedToSoftMuon',
 'ProbeTracks_genPartIdx',
 'nBToKEE',
 'nElectron',
 'nMuon',
 'nProbeTracks',
 'HLT_*',
 'nGenPart',    ## always load n for the vector length
 'GenPart_pdgId',
 'GenPart_pt',
 'GenPart_eta',
 'GenPart_phi',
 'GenPart_mass',
 'GenPart_genPartIdxMother',
]


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
    gen       = nf['GenPart']

    nprocessed += hlt.shape[0]
    # Attach the objects to the candidates
    electrons['genPdgId']    = gen[electrons['genPartIdx']].pdgId
    electrons['motherPdgId'] = gen[gen[electrons['genPartIdx']].genPartIdxMother].pdgId
    tracks['genPdgId']       = gen[tracks['genPartIdx']].pdgId

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
                  (bcands.k.isMatchedToEle == 0) & (bcands.k.isMatchedToSoftMuon == 0) & \
                  (bcands.fit_mass > 4.8) #&\
#                   (bcands.l_xy_sig > 1.)
#                   (bcands.fit_k_pt > 3) & (bcands.k.DCASig > 2) & (bcands.fit_pt > 3) & \
#                   (bcands.svprob > 0.1) & (bcands.fit_cos2D > 0.999) & \
        
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

    # df['trgmu_eta'] = sel_bcands.trg_mu.p4.eta
    # df['trgmu_pt'] = sel_bcands.trg_mu.p4.pt
    
    final_df = pd.concat((final_df, df))

# final_df.to_hdf(args.f_out, 'df', mode = 'w')
print('DONE! Processed events: ', nprocessed)
print('Saved events:', final_df.shape[0])


import numpy as np
out = uproot.recreate(f_out, compression = uproot.LZMA(8))

out['tree'] = uproot.newtree({'e1_pt'       : "float",
                              'e2_pt'       : "float",
                              'k_pt'        : "float",
                              'e1_phi'      : "float",
                              'e2_phi'      : "float",
                              'k_phi'       : "float",
                              'e1_eta'      : "float",
                              'e2_eta'      : "float",
                              'k_eta'       : "float",
                              'e1_isPF'     : "float",
                              'e2_isPF'     : "float",
                              'e1_pfOverlap': "float",
                              'e2_pfOverlap': "float",
                              'e1_mvaId'    : "float",
                              'e2_mvaId'    : "float",
                              'e1_unBDT'    : "float",
                              'e2_unBDT'    : "float",
                              'e1_dxyS'     : "float",
                              'e2_dxyS'     : "float",
                              'k_DCA'       : "float",
                              'k_dxyS'      : "float",
                              'B_charge'    : "float",
                              'B_mass'      : "float",
                              'B_mass_err'  : "float",
                              'B_pt'        : "float",
                              'B_eta'       : "float",
                              'B_phi'       : "float",
                              'B_svprob'    : "float",
                              'B_cos'       : "float",
                              'B_ls'        : "float",
                              'B_mll'       : "float",
                             })

out['tree'].extend({'e1_pt'       : np.array(df['e1_pt' ]),
                    'e2_pt'       : np.array(df['e2_pt' ]),
                    'k_pt'        : np.array(df['k_pt'  ]),
                    'e1_phi'      : np.array(df['e1_phi']),
                    'e2_phi'      : np.array(df['e2_phi']),
                    'k_phi'       : np.array(df['k_phi' ]),
                    'e1_eta'      : np.array(df['e1_eta']),
                    'e2_eta'      : np.array(df['e2_eta']),
                    'k_eta'       : np.array(df['k_eta' ]),
                    'e1_isPF'     : np.array(df['e1_isPF'     ]),
                    'e2_isPF'     : np.array(df['e2_isPF'     ]),
                    'e1_pfOverlap': np.array(df['e1_pfOverlap']),
                    'e2_pfOverlap': np.array(df['e2_pfOverlap']),
                    'e1_mvaId'    : np.array(df['e1_mvaId'    ]),
                    'e2_mvaId'    : np.array(df['e2_mvaId'    ]),
                    'e1_unBDT'    : np.array(df['e1_unBDT'    ]),
                    'e2_unBDT'    : np.array(df['e2_unBDT'    ]),
                    'e1_dxyS'     : np.array(df['e1_dxyS'     ]),
                    'e2_dxyS'     : np.array(df['e2_dxyS'     ]),
                    'k_DCA'       : np.array(df['k_DCA'      ]),
                    'k_dxyS'      : np.array(df['k_dxyS'     ]),
                    'B_charge'    : np.array(df['B_charge'   ]),
                    'B_mass'      : np.array(df['B_mass'     ]),
                    'B_mass_err'  : np.array(df['B_mass_err' ]),
                    'B_pt'        : np.array(df['B_pt'       ]),
                    'B_eta'       : np.array(df['B_eta'      ]),
                    'B_phi'       : np.array(df['B_phi'      ]),
                    'B_svprob'    : np.array(df['B_svprob'   ]),
                    'B_cos'       : np.array(df['B_cos'      ]),
                    'B_ls'        : np.array(df['B_ls'       ]),
                    'B_mll'       : np.array(df['B_mll'      ]),
                   })

# # out['tree'] = final_df
# # uproot.newtree({'a' : np.int32, 'b' : np.float32})
# # out["tree"].extend({'a' : np.array([1,2,3,4]), 'b' : np.array([1.1, 2.2, 3.3, 4.4])})
out.close()
