import os
import subprocess
import datetime
from optparse import OptionParser
import pdb
import math

parser = OptionParser()
parser.add_option("-e"  , "--era"    , dest = "era"    , help = "data era (2018B_p1,2018B_p2...)"                , default=  'JpsiEE_MC'                   )
parser.add_option("-n"  , "--njobs"  , dest = "njobs"  , help = "tot number of input files to be read. All = -1" , default = -1                            )
parser.add_option("-f"  , "--nfiles" , dest = "nfiles" , help = "choose number of files per job.Default is 1"    , default =  1                            )
parser.add_option("-c"  , "--chan"   , dest = "channel", help = "Kee, Kmm, ..."                                  , default = "Kee"                        )
parser.add_option("-d"  , "--outdir" , dest = "outdir" , help = "output dir"                                     , default = "default_folder"             )

parser.add_option("-t"  , "--test"   , dest = "test"   ,  help = "do not submit to queue"                        , default = False, action='store_true')
parser.add_option("--print"          , dest = "printN" ,  help = "print infos"                                   , default = False, action='store_true'      )

parser.add_option("-S"  , "--start"  , dest = "start"  , help = "choose starting file"                           , default =  0                            )
parser.add_option("-m"  , "--mc"     , dest = "mc"     , help = "is mc or data? mc = True, data = False"         , default=False, action='store_true'      )
# parser.add_option("-l"  , "--list"   , dest = "inputfiles" ,  help = "input file list"                           , default = "samples.py"                  )

(options,args) = parser.parse_args()  

eos_out_folder = '/eos/cms/store/cmst3/user/fiorendi/parking2018/ntuples/'

from samples import *

if options.printN:
    print '** output folder in eos:**\n', eos_out_folder, '\n'
    print '** possible samples: **\n', 
    for k,v in era_dict.items():
        print k, '\t', v 
    print '\n'
    exit()

key = datetime.datetime.strftime(
    datetime.datetime.now(), '%Y%h%d_%H%M'
)

## create file list based on the selected era
f_list_name = 'file_list_%s.txt'%options.era
os.system('ls %s/000*/*.root > %s'%(era_dict[options.era],f_list_name))
datapath = os.path.realpath(f_list_name)

## calculate n jobs to be submitted 
if options.njobs == -1:
    tot_n_files = int( os.popen('ls %s/000*/*.root | wc -l'%(era_dict[options.era])).read().strip())
else:    
    tot_n_files = int(options.njobs)

nf_per_job = int(options.nfiles)

n_jobs  = int(tot_n_files/float(nf_per_job))
# n_jobs = int(math.ceil(tot_n_files / nf_per_job ))


# name     = os.path.basename(options.script).split('.')[0]
name = 'skim_kee.py'

## local out folder for logs
newfolder = 'ntuples/' + options.outdir    
subprocess.check_call(['mkdir', newfolder]) 
subprocess.check_call(['mkdir', newfolder+'/scripts']) 
subprocess.check_call(['mkdir', newfolder+'/outCondor']) 

# import pdb; pdb.set_trace()
## eos output folder for root files
subprocess.check_call(['mkdir', eos_out_folder + newfolder.replace('ntuples/', '')]) 

## write batch.sh script
bname = newfolder + '/scripts/batch.sh'
with open(bname, 'w') as batch:
    batch.write('''#!/bin/tcsh

source /afs/cern.ch/user/f/fiorendi//miniconda3/etc/profile.d/conda.csh
conda deactivate

conda activate pyrk_env_sara
echo 'python /afs/cern.ch/work/f/fiorendi/private/parking/pyrk/pyrk/skim_kee.py $1 $2 $3 $4'
time python /afs/cern.ch/work/f/fiorendi/private/parking/pyrk/pyrk/skim_kee.py $1 $2 $3 $4
mv *.root {out} 
'''.format(out = eos_out_folder + newfolder.replace('ntuples/', ''))
)
subprocess.call(['chmod', '+x', bname])

# setenv PATH /afs/cern.ch/user/f/fiorendi//miniconda3/bin:$PATH

## write the cfg for condor submission condor_multiple_readnano.cfg
with open(newfolder + '/condor_sub.cfg', 'w') as cfg:
    cfg.write('''Universe = vanilla
Executable = {script}
use_x509userproxy = $ENV(X509_USER_PROXY)
Should_Transfer_Files = YES
WhenToTransferOutput = ON_EXIT
transfer_input_files = {datapath}
getenv = True
+AccountingGroup = "group_u_CMST3.all"
requirements = (OpSysAndVer =?= "CentOS7")
    
Log    = {out}/outCondor/condor_job_$(Process).log
Output = {out}/outCondor/condor_job_$(Process).out
Error  = {out}/outCondor/condor_job_$(Process).err
Arguments = {datapath} ntuple_{key}_{chan}_{era}_$(Process).root {njobs} $(Process)
Queue {njobs}
'''.format(script   = bname,
           datapath = f_list_name,
           out   = newfolder,
           key   = key,
           chan  = options.channel ,
           era   = options.era,
           njobs = n_jobs,
           ))
    
# +MaxRuntime = 160600
# environment = "LS_SUBCWD=/afs/cern.ch/work/f/fiorendi/private/parking/read_nano//eos/cms/store/user/fiorendi/parking/readnano/jpsimc_ee_kcvf"
# request_memory = 2000
# +JobFlavour = "longlunch"

# submit to the queue
print 'condor_submit {CFG}'.format(CFG = newfolder + '/condor_sub.cfg')
if not options.test:
    os.system("condor_submit {CFG}".format(CFG = newfolder + '/condor_sub.cfg'))   
