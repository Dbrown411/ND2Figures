from nd2reader import ND2Reader
import json, re
import numpy as np
from .findfiles import *
from app.utilities import *
from app.utilities import _check_all_folders_in_path


class ND2Accumulator:

    def __init__(self, group=None, identify=None, groupby=None):
        self._set_defaults()
        self.identify = identify
        self.groupby = groupby
        if group is not None:
            self.group = group

    ##Defaults/updating functions for mapping channels to proteins/colors
    def _set_defaults(self):
        self.channel_to_protein = {
            '405': 'DAPI',
            '488': '488',
            '561': '561',
            '640': '640'
        }
        self.channel_to_color = {
            "640": 'r',
            '561': 'r',
            '488': 'g',
            '405': 'b'
        }
        self.channel_standardization = {0: '405', 1: '488', 2: '561', 3: '640'}

    def set_sample_name(self, pattern):
        identifier_slice = self.identify if self.identify is not None else self.groupby if self.groupby is not None else (
            0, -1)
        sample_id = pattern[identifier_slice[0]:identifier_slice[1]]
        self.name = "_".join(sample_id)

    def standardize_channels(self, channels_dict):
        ##Standardize channel names
        channels = list(channels_dict.keys())
        standardization = {}
        for i, x in enumerate(channels):
            extracted_channel = None
            substr = re.search(r'\d{3}', x)
            if substr:
                extracted_channel = substr.group(0)
            if x in self.channel_to_protein.keys():
                c = x
            elif extracted_channel:
                wavelengths = [int(y) for y in self.channel_to_protein.keys()]
                c = str(closest(wavelengths, int(extracted_channel)))
            elif np.any([y in x for y in self.channel_to_protein.keys()]):
                for y in self.channel_to_protein.keys():
                    if y in c:
                        c = y
                        break
            else:
                c = self.channel_standardization[0]
            standardization[x] = c
        for k, v in standardization.items():
            if k != v:
                channels_dict[v] = channels_dict[k]
                del channels_dict[k]
        return channels_dict

    def merge_nd2_to_dict(self, grouped_images) -> "dict[str:list]":

        collected_channels_dicts = []
        for f in grouped_images:
            try:
                with ND2Reader(f.as_posix()) as images:
                    channels = images.metadata['channels']
                    numChannels = len(channels)
                    set_axes_to_iterate(images)
                    channels_dict = {
                        channels[i]: []
                        for i in range(numChannels)
                    }
                    for k, frame in enumerate(images):
                        channels_dict[channels[k % numChannels]].append(frame)
            except Exception as e:
                print(e)
                continue
            collected_channels_dicts.append((images.metadata, channels_dict))
        return collected_channels_dicts

    def merge_files(self, collected_channels_dicts):
        collected_channels_dicts.sort(key=lambda x: len(x[1]))
        merged = collected_channels_dicts.pop(0)
        meta1, merged = merged
        collected_meta = [meta1]
        for m, d in collected_channels_dicts:
            merged.update(d)
            collected_meta.append([m])
        return (collected_meta, merged)

    def _set_group(self, group):
        self.named_channels = {}
        identifier, folder, files = group
        self.folder = folder
        self.set_sample_name(identifier)
        collected_channels_dicts = self.merge_nd2_to_dict(files)

        ##get collection of data and merge
        self.metadata, channels_dict = self.merge_files(
            collected_channels_dicts)

        channels_dict = self.standardize_channels(channels_dict)

        channels = list(channels_dict.keys())
        _, first_channel = next(iter((channels_dict.items())))
        self.framesize = first_channel[0].shape

        channels.sort(reverse=True)
        self.channels = channels
        self.channel_order = {c: i for i, c in enumerate(channels)}
        self.named_channels = list(channels_dict.items())
        self.named_channels.sort(key=lambda x: x[0], reverse=True)

    @property
    def group(self):
        return self.named_channels

    @group.setter
    def group(self, grouped_images):
        if grouped_images is None:
            return
        self._set_group(grouped_images)

    @property
    def identify(self):
        return self._identify

    @identify.setter
    def identify(self, arr_slice):
        self._identify = arr_slice

    @property
    def groupby(self):
        return self._groupby

    @groupby.setter
    def groupby(self, arr_slice):
        self._groupby = arr_slice

    @property
    def folder(self):
        return self._folder

    @folder.setter
    def folder(self, folder):
        self._folder = folder
        self.get_channelmap(folder)

    @property
    def channels(self):
        return self._channels

    @channels.setter
    def channels(self, channels):
        self._channels = channels
        self.resolve_channels()

    def get_channelmap(self, folder):
        path_to_config = _check_all_folders_in_path(folder)
        if path_to_config is None:
            self.channel_to_protein = {
                "405": "405",
                "488": "488",
                "561": "561",
                "640": "640"
            }
        else:
            with open(path_to_config, 'r') as f:
                self.channel_to_protein = json.load(f)

    def resolve_channels(self):
        if len(self.channels) > 3:
            raise ValueError(
                'More then 3 color channels not supported for creating composite images'
            )
        if ('561' in self.channels) & ('640' in self.channels):
            self.channel_to_color = {
                "640": 'r',
                '561': 'g',
                '488': 'b',
                '405': 'b'
            }