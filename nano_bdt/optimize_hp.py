import os, sys
# sys.path.insert(0, os.environ['HOME'] + '/.local/lib/python2.7/site-packages')

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
# mpl.use('TkAgg')
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import pandas, root_numpy
from copy import deepcopy
import argparse
import pickle
import sklearn
from   sklearn.externals import joblib
from   sklearn.ensemble  import GradientBoostingClassifier
from   sklearn.metrics   import roc_curve
from   sklearn.model_selection import train_test_split
from   pdb import set_trace
from   array       import array

from   xgboost import XGBClassifier, plot_importance

from ROOT  import TFile, TTree, TH1F, gROOT, TChain
from math  import sqrt 

from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events

# mpl.interactive(True)

from utils.utils import *

mc_sigma = 0.06  ## average, by eye from fits
mc_mass  = BMass_ 

lumi_mc   = 3600 * (10./11)
lumi_data = 8.07 * (10./11)  ### more or less 

gROOT.SetBatch(True)


def xgboost_fom(max_depth,
                learning_rate,
                n_estimators,
                gamma,
                min_child_weight,
                max_delta_step,
                subsample,
                colsample_bytree,
                scale_pos_weight,
              ):

    clf = XGBClassifier(
        max_depth        = int(max_depth),
        learning_rate    = learning_rate,
        n_estimators     = int(n_estimators),
        subsample        = subsample,
        colsample_bytree = colsample_bytree,
        min_child_weight = min_child_weight,
        gamma            = gamma,     ## optimized
        max_delta_step   = max_delta_step,
#         reg_alpha        = 0,
#         reg_lambda       = 1,
        scale_pos_weight = scale_pos_weight,
        seed             = 1986,
        silent           = True,
        )
    
    clf.fit(
        train[features], 
        train.target,
        eval_set              = [(train[features], train.target), (test[features], test.target)],
        early_stopping_rounds = 100,
        eval_metric           = 'auc',
        verbose               = False,
        sample_weight         = train['normfactor'],
    )

    true_bkg = test[(test.target==0) ]
    true_sig = test[(test.target==1) ]
#     true_bkg = test[(test.target==0) & (test.pass_preselection==1)]
#     true_sig = test[(test.target==1) & (test.pass_preselection==1)]
    
    true_bkg['bdt_score']  = clf.predict_proba(true_bkg[features])[:, 1] ## should be probability to be signal (bdt_score) for each data entry
    true_sig['bdt_score']  = clf.predict_proba(true_sig[features])[:, 1] ## should be probability to be signal (bdt_score) for each data entry
    

    significances = {}
    print 'score \t S (on test only) \t B (on test s.) \t significance'
    for wp in np.arange(0.5,1,0.02):
        n_sig = float( len(true_sig[true_sig.bdt_score > wp])) * lumi_data/lumi_mc 
        n_bkg = float( len(true_bkg[true_bkg.bdt_score > wp])) * 6./8 ## to account that we use 8 sigma region for bkg and 5 sigma region for signal
        if n_sig > 0:
            significances[wp] =  n_sig/ sqrt(n_sig + n_bkg) 
        else:
            significances[wp] = 0.0
        
        print wp, '\t', n_sig, '\t', n_bkg, '\t', significances[wp]
    return max(significances.values())

#     return clf.evals_result()['validation_0']['auc'][-1]

#     return cross_val_score(xgb.XGBClassifier(max_depth=int(max_depth),
#                                              learning_rate=learning_rate,
#                                              n_estimators=int(n_estimators),
#                                              nthread=nthread,
#                                              subsample=subsample,
#                                              colsample_bytree=colsample_bytree),
#                            train,
#                            y,
#                            "roc_auc",
#                            cv=5).mean()


##########################################################################################
#####   SIGNAL AND BACKGROUND SELECTION
##########################################################################################
mc_match  = '(abs(e1_genPdgId)==11  && abs(e2_genPdgId)==11  && \
              abs(k_genPdgId) ==321                          && \
              ((e1_genMumId== 521 && e2_genMumId== 521 && k_genMumId==521) || \
               (e1_genMumId==-521 && e2_genMumId==-521 && k_genMumId==-521))) && '

### selection for B->Jpsiee
# mc_match  = '(abs(e1_genPdgId)==11  && abs(e2_genPdgId)==11  && \
#               e1_genMumId==443      && e2_genMumId==443      && \
#               abs(k_genPdgId) ==321                          && \
#               ((e1_genGMaId== 521 && e2_genGMaId== 521 && k_genMumId==521) || \
#                (e1_genGMaId==-521 && e2_genGMaId==-521 && k_genMumId==-521))) && '

pfoverlap = '(e1_pfOverlap == 0 && e2_pfOverlap == 0) && '

sig_selection_cutbased = mc_match + pfoverlap + '((B_mass > {M}-3.*{S} && B_mass < {M}+3.*{S}) && \
                                                  (B_mll < 2.45)) '.format( M=mc_mass,S=mc_sigma)
#                            trig == 0 && \

bkg_selection_cutbased = pfoverlap + '(((B_mass > {M}-7.*{S} && B_mass < {M}-3.*{S}) ||  \
                                        (B_mass > {M}+3.*{S} && B_mass < {M}+7.*{S})) && \
                                       (B_mll < 2.45)) '.format( M=mc_mass,S=mc_sigma)



sig_list = [
          'sub_samples/sample_2018_MC_Kee_0.root',
          'sub_samples/sample_2018_MC_Kee_1.root',
#           'sub_samples/sample_2018_MC_Kee_2.root',
          'sub_samples/sample_2018_MC_Kee_3.root',
          'sub_samples/sample_2018_MC_Kee_4.root',
          'sub_samples/sample_2018_MC_Kee_5.root',
          'sub_samples/sample_2018_MC_Kee_6.root',
          'sub_samples/sample_2018_MC_Kee_7.root',
          'sub_samples/sample_2018_MC_Kee_8.root',
          'sub_samples/sample_2018_MC_Kee_9.root',
          'sub_samples/sample_2018_MC_Kee_10.root',
]


bkg_list = [
          'sub_samples/sample_2018_data_Kee_0.root',
          'sub_samples/sample_2018_data_Kee_1.root',
#           'sub_samples/sample_2018_data_Kee_2.root',
          'sub_samples/sample_2018_data_Kee_3.root',
          'sub_samples/sample_2018_data_Kee_4.root',
          'sub_samples/sample_2018_data_Kee_5.root',
          'sub_samples/sample_2018_data_Kee_6.root',
          'sub_samples/sample_2018_data_Kee_7.root',
          'sub_samples/sample_2018_data_Kee_8.root',
          'sub_samples/sample_2018_data_Kee_9.root',
          'sub_samples/sample_2018_data_Kee_10.root',
]



tag = '_optimize'

##########################################################################################
#####   FEATURES AND BRANCHES
##########################################################################################
features = [
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
]

branches = features + [
    'B_mll',
    'B_mass',
    'e1_pfOverlap',
    'e2_pfOverlap',
]

branches_mc = [
    'e1_genPdgId',
    'e2_genPdgId',
    'e1_genMumId',
    'e2_genMumId',
    'k_genPdgId',
    'e1_genGMaId',
    'e2_genGMaId',
    'k_genMumId',
]
branches = list(set(branches))

sig = pandas.DataFrame(
    root_numpy.root2array(
        sig_list, 
        'tree',
        branches  = list(set(branches + branches_mc)) ,
        selection = sig_selection_cutbased,
    )
)

bkg = pandas.DataFrame(
    root_numpy.root2array(
        bkg_list, 
        'tree',
        branches  = branches,
        selection = bkg_selection_cutbased,
    )
)

##########################################################################################
#####   DEFINE THE TARGETS
##########################################################################################

sig['target'] = np.ones (sig.shape[0]).astype(np.int)
bkg['target'] = np.zeros(bkg.shape[0]).astype(np.int)

sig['normfactor'] = 1.
bkg['normfactor'] = 1. 


##########################################################################################
#####   SPLIT TRAIN AND TEST SAMPLES
##########################################################################################
data_all = pandas.concat([sig, bkg], sort=True)

## add column for pass-preselection
# data_all['pass_preselection'] = ( data_all.mumTMOneStationTight == 1 ) & ( data_all.mupTMOneStationTight == 1 ) & \
#                                 ( data_all.kkMass > 1.035 ) & \
#                                 (~((data_all.kstTrkmGlobalMuon == 1) & ( data_all.kstTrkmNTrkLayers > 5 ) & ( data_all.kstTrkmNPixHits > 0))) & \
#                                 (~((data_all.kstTrkpGlobalMuon == 1) & ( data_all.kstTrkpNTrkLayers > 5 ) & ( data_all.kstTrkpNPixHits > 0))) & \
#                                 (( (data_all.charge_trig_matched ==  1) & (data_all.kstTrkpPt > 1.2) & (data_all.trkpDCASign > 2) ) | \
#                                  ( (data_all.charge_trig_matched == -1) & (data_all.kstTrkmPt > 1.2) & (data_all.trkmDCASign > 2) ) )


# data = data_all[data_all.pass_preselection==1]
data = data_all
train, test = train_test_split(data, test_size=0.3, random_state = 17)

## weight for unbalanced datasets
ratio_pos_weight = bkg['B_mll'].count() / sig['B_mll'].count()

# set_trace()
xgboostBO = BayesianOptimization(xgboost_fom,
                                 {
                                  'max_depth': (2, 15),
                                  'learning_rate': (0.001, 0.4),
                                  'n_estimators': (200, 900),
                                  'subsample': (0.4, 0.8),
                                  'colsample_bytree' :(0.5, 0.99),
                                  'min_child_weight': (2, 12),
                                  'gamma': (0., 5.),
                                  'max_delta_step': (0, 0.4),
                                  'scale_pos_weight': (ratio_pos_weight,ratio_pos_weight)
                                  }
                                )
## save optimization steps
logger = JSONLogger(path="logs_firstOpt_retry.json")
xgboostBO.subscribe(Events.OPTMIZATION_STEP, logger)

xgboostBO.maximize(
    n_iter = 20,
    init_points = 30,
)

print('-'*53)

print('Final Results')
print(xgboostBO.max)

print '\n ---- all iterations ------ \n'
for i, res in enumerate(xgboostBO.res):
    print("Iteration {}: \n\t{}".format(i, res))