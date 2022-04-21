from pathlib import Path
from nd2reader import ND2Reader
import cv2 as cv2
import numpy as np


def write_proj(proj, path: Path, filename, ext='.tif'):
    outpath = path / f"{filename}{ext}"
    cv2.imwrite(outpath.as_posix(), proj)


def _check_all_folders_in_path(path: Path) -> Path:
    channel_map = path / "channelmap.txt"
    if not channel_map.is_file():
        try:
            _check_all_folders_in_path(path.parent)
        except:
            return None
    else:
        return channel_map


def normalize_frame(frame, max=255):
    return cv2.normalize(frame, None, 0, max, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def find_frame_maxes(frames=[]):
    norm_against = [np.max(x) for x in frames]
    overall_max = np.max(norm_against)
    norm_against = [
        int(np.round((np.divide(x, overall_max) * 255))) for x in norm_against
    ]
    return norm_against


def closest(lst, K):
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]


def set_axes_to_iterate(images: ND2Reader):
    iteraxes = []
    if int(images.metadata['total_images_per_channel']) > 1:
        iteraxes.append('z')
    if len(images.metadata['channels']) > 1:
        iteraxes.append('c')
    images.iter_axes = iteraxes