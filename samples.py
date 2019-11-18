# USERS dict, if is not there will default to 'default'

import os
users = {
    'mverzett' : {
        'out' : '/eos/cms/store/cmst3/user/mverzett/parking2018/',
        'cmg' : True,
    },
    'fiorendi' : {
        'out' : '/eos/cms/store/cmst3/user/fiorendi/parking2018/',
        'cmg' : True,
    },
    'default' : {
        'out' : f'/eos/cms/store/group/cmst3/group/bpark/{os.environ["USER"]}',
        'cmg' : False,
    },
}

genericPath = '/eos/cms/store/group/cmst3/group/bpark/'
samples = {
    ## MC
    'BJpsiK_ee_mc_2019Oct25' : {
        'path' : f'{genericPath}/BParkingNANO_2019Oct25/BuToKJpsi_Toee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BuToKJpsi_Toee/191025_125913/',
        'isMC' : True,
        'splitting' : 5,
    },
    'BJpsiKst_ee_mc_2019Oct25' : {
        'path' : f'{genericPath}/BParkingNANO_2019Oct28/BdToKstarJpsi_ToKPiee_Mufilter_SoftQCDnonD_TuneCP5_13TeV-pythia8-evtgen/crab_BdToKstarJpsi_Toee/191028_080830/',
        'isMC' : True,
        'splitting' : 5,
    },

    ## DATA
    '2018A_BPH2_2019Oct21' : {
        'path' : f'{genericPath}/BParkingNANO_2019Oct21/ParkingBPH2/crab_data_Run2018A_part2/191021_131326/',
        'isMC' : False,
        'splitting' : 10,
    },
    '2018B_BPH2_2019Oct21' : {
        'path' : f'{genericPath}/BParkingNANO_2019Oct21/ParkingBPH2/crab_data_Run2018B_part2/191021_131046/',
        'isMC' : False,
        'splitting' : 10,
    },
}
