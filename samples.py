era_dict = {}
n_dict   = {}

genericPath       = '/eos/cms/store/group/cmst3/group/bpark/'
genericPath_storm = 'root://xrootd-cms.infn.it//store/user/fiorendi/p5prime/'

## MC ##
pathBJpsiK_ee_mc = 'BParkingNANO_2019Oct25/BuToKJpsi_Toee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKJpsi_Toee/191025_125913/'
era_dict['BJpsiK_ee_mc'] = genericPath + pathBJpsiK_ee_mc

pathBJpsiKst_ee_mc = 'BParkingNANO_2019Oct28/BdToKstarJpsi_ToKPiee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BdToKstarJpsi_Toee/191028_080830/'
era_dict['BJpsiKst_ee_mc'] = genericPath + pathBJpsiKst_ee_mc



## data ##
path18A_BPH2 = 'BParkingNANO_2019Oct21/ParkingBPH2/crab_data_Run2018A_part2/191021_131326'
era_dict['2018A_BPH2'] = genericPath + path18A_BPH2

path18B_BPH2 = 'BParkingNANO_2019Oct21/ParkingBPH2/crab_data_Run2018B_part2/191021_131046'
era_dict['2018B_BPH2'] = genericPath + path18B_BPH2




# n_dict['JpsiEE_MC']   = 59


# pathJpsiEE_MC_KCVF = 'BParkingNANO_2019Oct15/BuToKJpsi_Toee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKJpsi_Toee/191015_112059/'
# era_dict['JpsiEE_MC_KCVF'] = [ genericPath + pathJpsiEE_MC_KCVF]
# n_dict['JpsiEE_MC_KCVF']   = 74
# 
# 
# pathNREE_MC = 'BParkingNANO_2019Sep12/BuToKee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKee/190912_155644/'
# era_dict['NREE_MC'] = [ genericPath + pathNREE_MC]
# n_dict['NREE_MC']   = 7
# 
# pathJpsiMuMu_MC = 'BParkingNANO_2019Sep12/BuToKJpsi_ToMuMu_probefilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKJpsi_ToMuMu/190912_155525/'
# era_dict['JpsiMuMu_MC'] = [ genericPath + pathJpsiMuMu_MC]
# n_dict['JpsiMuMu_MC']   = 59
# 
# 
# 
# ##  data ##
# 
# n_dict['2018B_BPH2']   = 1349
# 
# path18C_BPH2 = 'BParkingNANO_2019Sep12/ParkingBPH2/crab_data_Run2018C_part2/190912_155245'
# era_dict['2018C_BPH2'] = [ genericPath + path18C_BPH2]
# n_dict['2018C_BPH2']   = 1458
# 
# path18D_BPH2 = 'BParkingNANO_2019Sep12/ParkingBPH2/crab_data_Run2018D_part2/190912_155004'
# era_dict['2018D_BPH2'] = [ genericPath + path18D_BPH2]
# n_dict['2018D_BPH2']   = 7345
# 
# 
# 
# path18A_BPH3 = 'BParkingNANO_2019Sep12/ParkingBPH3/crab_data_Run2018A_part3/190912_154846'
# era_dict['2018A_BPH3'] = [ genericPath + path18A_BPH2]
# n_dict['2018A_BPH3']   = 1353
# 
# path18B_BPH3 = 'BParkingNANO_2019Sep12/ParkingBPH3/crab_data_Run2018B_part3/190912_183750'
# era_dict['2018B_BPH3'] = [ genericPath + path18B_BPH2]
# n_dict['2018B_BPH3']   = 1352
# 
# path18C_BPH3 = 'BParkingNANO_2019Sep12/ParkingBPH3/crab_data_Run2018C_part3/190912_155407'
# era_dict['2018C_BPH3'] = [ genericPath + path18C_BPH2]
# n_dict['2018C_BPH3']   = 1469
