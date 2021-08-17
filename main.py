from nd2reader import ND2Reader
import matplotlib as mpl
mpl.use("Qt5Agg")
import matplotlib.pyplot as plt
import glob,os
import numpy as np
import cv2 as cv2

def framestack_to_vol(framestack):
    volume_shape = (len(framestack),framestack[0].shape[0],framestack[0].shape[1])
    vol = np.empty(volume_shape,dtype=np.uint16)
    for i,frame in enumerate(framestack):
        vol[i,:,:] = frame
    return vol

def project_images(framestack):
    vol = framestack_to_vol(framestack)
    AIP = np.nanmean(vol[0:,:,:],axis=0,dtype=np.uint16)
    return AIP

def write_figure(fig,suffix=''):
    cv2.imwrite(f"{outname}-AIP{suffix}{ext}", fig)

directory = r"D:\OneDrive - Georgia Institute of Technology\Lab\Data\IHC\Confocal\Eyes\ACAN\01 - Control\Dual Stain Poly + Mono ACAN\Raw\MACAN and PNitige"
for f in glob.glob(directory+os.sep+'*.nd2'):
    outname= f[:-4]
    ext = '.tif'
    with ND2Reader(f) as images:
        channels = images.metadata['channels']
        numChannels = len(channels)

        channels_dict = {i:[] for i in range(numChannels)}
        images.iter_axes = ['z','c'][:-int(numChannels<2)]
        for i,frame in enumerate(images):
            channels_dict[i%numChannels].append(frame)
        for k,v in channels_dict.items():
            proj = project_images(v)
            write_figure(proj,suffix=f'-C{k}')




