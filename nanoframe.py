import awkward as awk
from coffea.analysis_objects import JaggedCandidateArray
import uproot
import uproot_methods

class NanoFrame():
    'Simple class that provides a lazy interface with the NanoAODs'
    def __init__(self, infile):
        self.uf = uproot.open(infile)
        self.tt = self.uf['Events']
        self.keys_ = set([i.decode() for i in self.tt.keys()])
        self.cache_ = set()
        self.table_ = awk.Table()

    def __getitem__(self, key):
        if key in self.cache_:
            return self.table_[key]
        elif key in self.keys_:
            ret = self.tt.array(key)
            self.table_[key] = ret
            self.cache_.add(key)
            return self.table_[key]
        else:
            branch = key + '_'
            subset = [k for k in self.keys_ if k.startswith(branch)]
            info = {i.replace(branch, '') : self.tt.array(i) for i in subset}
            counter = 'n' + key
            counts = 0
            if counter in self.keys_:
                counts = self.tt.array(counter)
                for name, branch in info.items():
                    if not (branch.count() == counts).all():
                        raise ValueError(f'Key {name} does not have the right shape')
                info = {i : j.content for i, j in info.items()}
            #check that everything is there to make a p4
            if all(i in info for i in ['pt', 'eta', 'phi', 'mass']): 
                ret = JaggedCandidateArray.candidatesfromcounts(
                    counts,
                    **info
                )
            else:
                ret = awk.Table(**info)
                if all(i in info for i in ['pt', 'phi']): 
                    ret['p4'] = uproot_methods.TLorentzVectorArray.from_ptetaphi(
                        ret['pt'], 0, ret['phi'], 0
                    )
            
            self.table_[key] = ret
            self.cache_.add(key)
            return self.table_[key]

    def __setitem__(self, key, val):
        self.table_[key] = val

    @property
    def objects(self):
        'branch groups available'
        return list(
            set(
                i.split('_')[0] for i in self.keys_ 
                if '_' in i
            )
        )
        
    @property
    def keys(self):
        'Keys available for loading and usage'
        return list(self.keys_)
    
    @property
    def columns(self):
        'columns already loaded'
        return self.table_.columns
