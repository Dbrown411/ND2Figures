from nd2reader import ND2Reader
import os, json, re
import numpy as np

def closest(lst, K):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

def _check_all_folders_in_path(path):
    path_to_config = f"{path}{os.sep}channelmap.txt"
    if not os.path.isfile(path_to_config):
        try:
            _check_all_folders_in_path(os.path.split(path_to_config)[0])
        except:
            return None
    else:
        return path_to_config

class ND2Accumulator:
    def __init__(self,group=None,identify=None,groupby=None):
        self._set_defaults()
        self.identify = identify
        self.groupby = groupby
        if group is not None:
            self.group = group

    ##Defaults/updating functions for mapping channels to proteins/colors
    def _set_defaults(self):
        self.channel_to_protein = {
                        '405':'DAPI',
                        '488':'488',
                        '561':'561',
                        '640':'640'
                        }
        self.channel_to_color = {
                        "640":'r',
                        '561':'r',
                        '488':'g',
                        '405':'b'
                        }
        self.channel_standardization = {
            0:'405',
            1:'488',
            2:'561',
            3:'640'
        }

    def set_sample_name(self,pattern):
        identifier_slice = self.identify if self.identify is not None else self.groupby if self.groupby is not None else (0,-1)
        sample_id = pattern[identifier_slice[0]:identifier_slice[1]]
        self.name = "_".join(sample_id)
    def standardize_channels(self,channels_dict):
        ##Standardize channel names
        channels = list(channels_dict.keys())
        standardization = {}
        for i,x in enumerate(channels):
            extracted_channel = None
            substr = re.search(r'\d{3}', x)
            if substr:
                extracted_channel = substr.group(0)
            if x in self.channel_to_protein.keys():
                c = x
            elif extracted_channel:
                wavelengths = [int(y) for y in self.channel_to_protein.keys()]
                c = str(closest(wavelengths,int(extracted_channel)))
            elif np.any([y in x for y in self.channel_to_protein.keys()]):
                for y in self.channel_to_protein.keys():
                    if y in c:
                        c = y
                        break
            else:
                c = self.channel_standardization[i]
            standardization[x] = c
        for k,v in standardization.items():
            channels_dict[v] = channels_dict[k]
            del channels_dict[k]
        return channels_dict
    def merge_nd2_to_dict(self,grouped_images):
        def set_axes_to_iterate(images):
            iteraxes = []
            if int(images.metadata['total_images_per_channel'])>1:
                iteraxes.append('z')
            if len(images.metadata['channels'])>1:
                iteraxes.append('c')
            images.iter_axes = iteraxes

        collected_channels_dicts = []
        for i,(_,pattern,f) in enumerate(list(grouped_images[1])):
            if i==0:
                self.set_sample_name(pattern)
            print(f)
            with ND2Reader(f) as images:
                # print(images.metadata)
                channels = images.metadata['channels']
                numChannels = len(channels)
                set_axes_to_iterate(images)
                channels_dict = {channels[i]:[] for i in range(numChannels)}
                for k,frame in enumerate(images):
                    channels_dict[channels[k%numChannels]].append(frame)
            collected_channels_dicts.append((images.metadata, channels_dict))
        return collected_channels_dicts

    def merge_files(self,collected_channels_dicts):
        collected_channels_dicts.sort(key=lambda x: len(x[1]))
        merged = collected_channels_dicts.pop(0)
        meta1,merged = merged
        collected_meta = [meta1]
        for m,d in collected_channels_dicts:
            merged.update(d)
            collected_meta.append([m])
        return (collected_meta, merged)

    def set_group(self,grouped_images):
        self.named_channels = {}
        collected_channels_dicts = self.merge_nd2_to_dict(grouped_images)
        
        ##get collection of data and merge
        self.metadata, channels_dict = self.merge_files(collected_channels_dicts)

        channels_dict = self.standardize_channels(channels_dict)
        

        channels = list(channels_dict.keys())
        _,first_channel = next(iter((channels_dict.items())))
        self.framesize = first_channel[0].shape

        channels.sort(reverse=True)
        self.channels = channels
        self.channel_order = {c:i for i,c in enumerate(channels)}
        self.named_channels = list(channels_dict.items())
        self.named_channels.sort(key=lambda x: x[0],reverse=True)
        print('Recognized Channels:')
        [print(f"{x}nm") for x in channels]
        print('')

    @property
    def group(self):
        return self.named_channels
    @group.setter
    def group(self,grouped_images):
        if grouped_images is None:
            return
        self.set_group(grouped_images)
        
    @property
    def identify(self):
        return self._identify
    @identify.setter
    def identify(self,arr_slice):
        self._identify = arr_slice

    @property
    def groupby(self):
        return self._groupby
    @groupby.setter
    def groupby(self,arr_slice):
        self._groupby = arr_slice

    @property
    def folder(self):
        return self._folder
        
    @folder.setter
    def folder(self,folder):
        self._folder = folder
        self.get_channelmap(folder)
        print(folder)
        print(self.channel_to_protein)
    
    @property
    def channels(self):
        return self._channels
    @channels.setter
    def channels(self,channels):
        self._channels = channels
        self.resolve_channels()

    def get_channelmap(self,folder):
        path_to_config = _check_all_folders_in_path(folder)
        if path_to_config is None:
            try:
                with open('default_channelmap.txt','r') as f:
                    self.channel_to_protein = json.load(f)
            except Exception as e:
                print(e)
                pass
        else:
            with open(path_to_config,'r') as f:
                self.channel_to_protein = json.load(f)
    
    def resolve_channels(self):
        if len(self.channels)>3:
            raise ValueError('More then 3 color channels not supported for creating composite images')
        if ('561' in self.channels)&('640' in self.channels):
            self.channel_to_color = {
                                    "640":'r',
                                    '561':'g',
                                    '488':'b',
                                    '405':'b'
                                    }
            pass
    