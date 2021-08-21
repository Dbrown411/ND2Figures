import os,glob
import itertools as it
import re as r

##Data retrieval/organization
def create_folders(folder,export_flags,clear=False):
    paths = {
        'raw':(export_flags['raw'],f"{folder}{os.sep}RawTifStacks"),
        'proj':(export_flags['proj'],f"{folder}{os.sep}16bitProjections"),
        'figure':(export_flags['figure'],f"{folder}{os.sep}Figures")
    }
    paths_out = []
    for type,(use,p) in paths.items():
        paths_out.append(p)
        if use:
            try:
                os.mkdir(p)
            except FileExistsError:
                if clear:
                    for file in os.scandir(p):
                        os.remove(file.path)
                else:
                    pass
            except Exception as e:
                print(e)
                pass

    return paths_out

def scantree(p):
    yielded = set()
    def _scantree(path):
        """Recursively yield DirEntry objects for given directory."""
        for entry1 in os.scandir(path):
            if entry1.name[0]=='.':
                continue
            if entry1.is_dir(follow_symlinks=False):
                for entry2 in _scantree(entry1.path):
                    yield entry2
            else:
                if (path in yielded):
                    continue
                if (entry1.name[-4:]!='.nd2'):
                    continue
                else:
                    yielded.add(path)
                    yield path
    return _scantree(p)

def scan_data(datapath):
    nd2Locations = list(scantree(datapath))
    return sorted(nd2Locations, key=lambda s: os.path.basename(s),reverse=True)

def match_scans(folder,groupby=(0,2)):
    nd2Files = list(glob.glob(folder+os.sep+'*.nd2'))
    if groupby is None:
        return [(os.path.split(file)[1].replace(' ','').upper(),[(i,os.path.split(file)[1].replace(' ','').upper(),file)]) for i, file in enumerate(nd2Files)]
    nd2Filenames = [(i,r.split(r'-|_|\.',os.path.split(file)[1].replace(' ','').upper()),file) for i, file in enumerate(nd2Files)]
    grouped = it.groupby(nd2Filenames,key=lambda x: x[1][groupby[0]:groupby[1]])
    return grouped