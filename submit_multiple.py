#! /bin/env python

import os
import subprocess
import datetime
from argparse import ArgumentParser
import pdb
import math
from samples import samples, users
from glob import glob
from pdb import set_trace

parser = ArgumentParser()
parser.add_argument("analyzer", help = "which analyser to run", default = 'BJpsiK_ee_mc' )
parser.add_argument("samples", help = "samples", nargs = '+', choices = samples.keys(), default = 'BJpsiK_ee_mc_2019Oct25' )
parser.add_argument("-n"  , "--njobs"  , dest = "njobs"  , type = int, help = "tot number of input files to be read. All = -1" , default = -1                            )
parser.add_argument("-d"  , "--outdir" , dest = "outdir" , help = "output dir"                                     , default = "ntuples" )
parser.add_argument("-t"  , "--test"   , dest = "test"   ,  help = "do not submit to queue"                        , default = False, action='store_true')
parser.add_argument("--print"          , dest = "printN" ,  help = "print infos"                                   , default = False, action='store_true'      )
parser.add_argument("-S"  , "--start"  , dest = "start"  , help = "choose starting file"                           , default =  0                            )
parser.add_argument("-f"  , "--flavour", help = "job flavour (sets running time) https://indico.cern.ch/event/731021/contributions/3013463/attachments/1656036/2651022/18-05-24_HTCondor_CMG.pdf", default =  'microcentury', choices = ['espresso', 'microcentury', 'longlunch', 'workday', 'tomorrow', 'testmatch', 'nextweek'])
args = parser.parse_args()

## Env variables
user_info = users.get(os.environ["USER"], users['default'])
eos_out_folder = user_info['out']
cmg_accounting = '+AccountingGroup = "group_u_CMST3.all"' if user_info['cmg'] else ''

conda_loc = os.environ['CONDA_EXE'].replace('bin/conda', 'etc/profile.d/conda.csh')
conda_env = os.environ['CONDA_DEFAULT_ENV']

script_loc = os.path.realpath(args.analyzer)
channel = os.path.basename(args.analyzer).replace('.py', '')

if args.printN:
    print('** output folder in eos:**\n', eos_out_folder, '\n')
    print('** possible samples: **\n') 
    for k,v in samples.items():
        print(k, '\t', v['path'])
    print('\n')
    exit()

key = datetime.datetime.strftime(
    datetime.datetime.now(), '%Y%h%d_%H%M'
)

for sample_name in args.samples:
    sample = samples[sample_name]
    ## local out folder for logs
    base_out = f'{args.outdir}/{channel}/{sample_name}'
    os.makedirs(f'{base_out}/scripts')
    os.makedirs(f'{base_out}/outCondor')
    
    ## eos output folder for root files
    full_eos_out = f'{eos_out_folder}/{base_out}/'
    os.makedirs(full_eos_out) 
    
    ## create file list based on the selected era
    f_list_name = f'{base_out}/scripts/file_list_{sample_name}.txt'
    flist = glob(f'{sample["path"]}/000*/*.root')
    with open(f_list_name, 'w') as f_list:
        f_list.write(
            '\n'.join(
                flist
                )
            )
    datapath = os.path.realpath(f_list_name)
    
    ## calculate n jobs to be submitted 
    if args.njobs == -1:
        tot_n_files = len(flist)
    else:    
        tot_n_files = args.njobs
    
    n_jobs  = int(tot_n_files/float(sample['splitting']))
    
    ## write batch.sh script
    bname = os.path.realpath(f'{base_out}/scripts/batch.sh')
    mc_flag = '--mc' if sample.get('isMC', False) else ''
    with open(bname, 'w') as batch:
        batch.write(f'''#!/bin/tcsh
    
source {conda_loc}
conda deactivate

conda activate {conda_env}
echo "python {script_loc} $1 $2 $3 $4 {mc_flag}"
time python {script_loc} $1 $2 $3 $4 {mc_flag}
mv *.root {full_eos_out} 
''')
    subprocess.call(['chmod', '+x', bname])
    
    ## write the cfg for condor submission condor_multiple_readnano.cfg
    with open(f'{base_out}/condor_sub.cfg', 'w') as cfg:
        cfg.write(f'''Universe = vanilla
Executable = {bname}
use_x509userproxy = $ENV(X509_USER_PROXY)
Should_Transfer_Files = YES
WhenToTransferOutput = ON_EXIT
getenv = True
+JobFlavour = "{args.flavour}"
{cmg_accounting}
requirements = (OpSysAndVer =?= "CentOS7")
    
Log    = {base_out}/outCondor/condor_job_$(Process).log
Output = {base_out}/outCondor/condor_job_$(Process).out
Error  = {base_out}/outCondor/condor_job_$(Process).err
Arguments = {datapath} ntuple_{key}_{channel}_{sample_name}_$(Process).root {n_jobs} $(Process)
Queue {n_jobs}
        ''')
        
    # +MaxRuntime = 160600
    # environment = "LS_SUBCWD=/afs/cern.ch/work/f/fiorendi/private/parking/read_nano//eos/cms/store/user/fiorendi/parking/readnano/jpsimc_ee_kcvf"
    # request_memory = 2000
    # +JobFlavour = "longlunch"
    
    # submit to the queue
    print('condor_submit {CFG}'.format(CFG = f'{base_out}/condor_sub.cfg'))
    if not args.test:
        os.system("condor_submit {CFG}".format(CFG = f'{base_out}/condor_sub.cfg'))   
