import os
import itertools as it
import re as r
from pathlib import Path
import attr


@attr.s(slots=True, kw_only=True)
class ExportLocation:
    raw: Path = attr.ib(default='')
    proj: Path = attr.ib(default='')
    fig: Path = attr.ib(default='')
    offset: Path = attr.ib(default='')


##Data retrieval/organization
def create_folders(folder: Path,
                   export_flags: dict,
                   clear: bool = False) -> "list[Path]":
    output_folder = folder / "Output"

    paths = ExportLocation(
        raw=output_folder / "RawTifStacks" if export_flags['raw'] else '',
        proj=output_folder /
        "16bitProjections" if export_flags['proj'] else '',
        fig=output_folder / "Figures" if export_flags['figure'] else '',
        offset=output_folder /
        "OffsetCorrection" if export_flags['offset'] else '',
    )

    for _, p in attr.asdict(paths).items():
        if p != '':
            try:
                p.mkdir()
            except FileExistsError:
                if clear:
                    for file in os.scandir(p.as_posix()):
                        os.remove(file.path)
                else:
                    pass
            except Exception as e:
                print(e)
                pass

    return paths


def scantree(p: Path):
    yielded = set()

    def _scantree(path: str):
        """Recursively yield DirEntry objects for given directory."""
        for entry1 in os.scandir(path):
            if entry1.name[0] == '.':
                continue
            if entry1.is_dir(follow_symlinks=False):
                for entry2 in _scantree(entry1.path):
                    yield entry2
            else:
                if (path in yielded):
                    continue
                if (entry1.name[-4:] != '.nd2'):
                    continue
                else:
                    yielded.add(path)
                    yield path

    return _scantree(p.as_posix())


def scan_data(datapath: Path):
    nd2Locations = map(Path, list(scantree(datapath)))
    return sorted(nd2Locations, key=lambda s: s.stem, reverse=True)


def match_scans(folder: Path, groupby=(0, 2)):
    nd2Files = list(folder.glob('*.nd2'))
    normalize = lambda x: str(x).replace(' ', '').upper()
    tokenize = lambda x: r.split(r'-|_|\.', normalize(x.stem))

    keyfunc = lambda x: tokenize(x)[groupby[0]:groupby[1]]

    if groupby is None:
        return [(tokenize(file), file) for file in nd2Files]
    grouped = it.groupby(sorted(nd2Files, key=keyfunc), key=keyfunc)
    out = []
    for k, g in grouped:
        out.append((k, folder, list(g)))
    return out
