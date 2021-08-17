from nd2reader import ND2Reader
import matplotlib as mpl
mpl.use("Qt5Agg")
import matplotlib.pyplot as plt
import glob,os
import numpy as np
import cv2 as cv2

def project_images(framestack,write=True,suffix=''):
    volume_shape = (len(framestack),framestack[0].shape[0],framestack[0].shape[1])
    vol = np.empty(volume_shape,dtype=np.uint16)
    for i,frame in enumerate(framestack):
        vol[i,:,:] = frame
    AIP = np.nanmean(vol[0:,:,:],axis=0,dtype=np.uint16)
    if write:
        cv2.imwrite(f"{outname}-AIP{suffix}{ext}", AIP)

directory = r"D:\OneDrive - Georgia Institute of Technology\Lab\Data\IHC\Confocal\Eyes\ACAN\01 - Control\Dual Stain Poly + Mono ACAN\Raw\MACAN and PNitige"
for f in glob.glob(directory+os.sep+'*.nd2'):
    outname= f[:-4]
    ext = '.tif'
    with ND2Reader(f) as images:
        try:
            channels = images.metadata['channels']
            channels_dict = {i:[] for i in range(len(channels))}
            images.iter_axes = ['z','c']
            for i,frame in enumerate(images):
                channels_dict[i%len(channels)].append(frame)
            for k,v in channels_dict.items():
                project_images(v,suffix=f'-C{k}')
        except Exception as e:
            print(e)

            project_images(images)


