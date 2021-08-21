## -*- coding: utf-8 -*-
"""ND2Figures
Version 1
Dillon Brown, 19Aug2021
"""
from plotting import *
from nd2merger import *
from findfiles import *

import os,argparse
import imageio
import numpy as np
import cv2 as cv2
from skimage import restoration

import matplotlib as mpl

"""
Functions/Use

1) Recursively crawl parent directory,
    find all .nd2 file types,
    optional compilation of different files using groupby
        --groupby tup(i,j) of filename split by (-|_|.)

2) (Optional) export a 16bit .tiff stack of experiment 

3) (Optional) Identify metadata and create/export 16bit projections 
                with optional filtering/thresholding
                    -By default assumes DAPI as 405nm
                    and creates a maximum intensity projection for more visible
                    counterstain
                    -other hardcoded channels (480nm,561nm,640nm) will be 
                    mean intensity projections more suitable for quantification

4) (Optional) Convert projections to normalized 8bit RGB images,
                map wavelengths to colors and labeled proteins,
                Plot in order of descending wavelength, and add
                a composite image at the end.
                Add scalebar on first image based on metadata
                Add labels for each channel on each image
                Size figure correctly in order to export figure at original resolution for each channel
                    if aspect ratio is >1: 
                        Arranges panels vertically
                    if aspect ratio is <=1: 
                        Arranges horizontally

"""







##Data functions (operating on u16)

def framestack_to_vol(framestack):
    volume_shape = (len(framestack),framestack[0].shape[0],framestack[0].shape[1])
    vol = np.empty(volume_shape,dtype=np.uint16)
    for i,frame in enumerate(framestack):
        vol[i,:,:] = frame
    return vol

def get_projection(vol,method='mean'):
    meth_map = {'mean':lambda x: np.nanmean(x[0:,:,:],axis=0,dtype=np.uint16),
            'med':lambda x: np.nanmedian(x[0:,:,:],axis=0),
            'max':lambda x: np.nanmax(x[0:,:,:],axis=0),
            'sum':lambda x: np.nansum(x[0:,:,:],axis=0,dtype=np.uint16)
    }
    return meth_map[method](vol)

def plot_result(image, background):
    fig, ax = plt.subplots(nrows=1, ncols=3)

    ax[0].imshow(normalize_frame(background), cmap='Greens')
    ax[0].set_title('Background')
    ax[0].axis('off')

    ax[1].imshow(normalize_frame(image), cmap='Greens')
    ax[1].set_title('Original image')
    ax[1].axis('off')


    ax[2].imshow(normalize_frame(image - background), cmap='Greens')
    ax[2].set_title('Result')
    ax[2].axis('off')

    fig.tight_layout()

def filter_vol(vol):
    radz = 5
    radxy = 5
    background = restoration.rolling_ball(
                                            vol,
                                            kernel=restoration.ellipsoid_kernel(
                                                (1, radxy*2, radxy*2),
                                                radxy*2
                                                ),
                                            nansafe=True
                                            )
    for i in range(vol.shape[0]):
        plot_result(vol[i, :,:], background[i, :,:])
        plt.show()
    return vol-background

def offset_projection(proj,new_0):
    original = np.array(proj, dtype=np.int32)
    max16 = np.max(original)
    newmax16 = max16-new_0
    newImage = np.array(original - new_0)
    newImage = newImage.clip(min=0)
    offset_proj = cv2.normalize(newImage, None, 0, newmax16, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    return offset_proj

def identify_foreground(img):
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((5,5),np.uint8)    
    th4 = cv2.dilate(th3,kernel,iterations = 5)
    th4 = cv2.morphologyEx(th4, cv2.MORPH_CLOSE, kernel,iterations=10)
    th4 = cv2.dilate(th4,kernel,iterations = 5*int(max(img.shape)/512))
    return th4

##Main data analysis pipeline
def analyze_fstack(vol,proj_type,calc_proj):
    if not calc_proj:
        return

    proj = get_projection(vol,proj_type)
    frame = normalize_frame(proj)

    mask = identify_foreground(frame)
    foreground = cv2.bitwise_and(frame, mask)
    mask = cv2.bitwise_not(mask)
    display_background = cv2.bitwise_and(frame, mask)

    mask16 = cv2.normalize(mask, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    background = cv2.bitwise_and(proj, mask16)
    mean_background = np.mean(background)
    offset = int(round(mean_background))
    if proj_type=='MAX':
        offset*=2
    offset_proj  = offset_projection(proj,offset)
    return offset_proj, (foreground,display_background)


def write_proj(proj,path,filename,ext='.tif'):
    cv2.imwrite(f"{path}{os.sep}{filename}{ext}", proj)


    
def main(directory,clear=True,groupby=None,identify=None,disp=False):
    if disp:
        mpl.use("Qt5Agg")
    else:
        mpl.use("Agg")
    import matplotlib.pyplot as plt
    

    foldersWithData = scan_data(directory)

    ##Clear previous results
    if compile_all:
        raw_path,proj_path, fig_path = create_folders(directory,export_flags,clear=clear)
    elif clear:
        [create_folders(folder,export_flags,clear=True) for folder in foldersWithData]

    plotter = SamplePlotter()
    folder_count = 0
    for folder in foldersWithData:
        folder_count+=1

        if not compile_all:
            raw_path,proj_path, fig_path = create_folders(folder,export_flags)
        file_groups = match_scans(folder,groupby = groupby)
        print('--'*10)
        print('--'*10)
        print(f'Folder {folder_count}:\n{folder}')
        print('--'*10)

        for group in file_groups:
            sample = ND2Accumulator(group,identify,groupby)
            sample.folder = folder
            plotter.fig_folder = fig_path
            plotter.sample = sample
            named_channels = sample.named_channels
            for i,(c,fstack) in enumerate(named_channels):
                projection_type = channel_to_proj_map[c]
                channel_outname = f'{sample.name}-{c}nm-{projection_type.upper()}'
                if export_flags['raw']:
                    imageio.mimwrite(f'{raw_path}{os.sep}{channel_outname}.tiff',fstack)
                vol = framestack_to_vol(fstack)
                ##Point at which custom analysis code could be easily injected
                ##Function should take a 3d np.vol of uint16 and return a 2d array for display
                ##
                proj, fgbg = analyze_fstack(vol,projection_type,calc_proj)


                if export_flags['proj']:
                    write_proj(proj,proj_path,channel_outname)
                if create_fig:
                    if not calc_proj:
                        proj, fgbg = analyze_fstack(fstack,projection_type,calc_proj)

                    plotter.set_channel(c,proj)

            if create_fig:
                plotter.plot_composite()
                if disp:
                    plt.show()
                if export_flags['figure']:
                    plotter.save_fig()

            plt.close()

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--clear', help="Clear previous exports",
                                    default=False,
                                    action='store_true')

    parser.add_argument('--show', help="show plots",
                                    default=False,
                                    action='store_true')

    parser.add_argument('--default', help="use default path",
                                    default=False,
                                    action='store_true')
    args = parser.parse_args()                     

    desktop = rf"D:"
    directory = rf"{desktop}{os.sep}OneDrive - Georgia Institute of Technology\Lab\Data\IHC\Confocal\Automated"
    if not args.default:
        from tkinter import filedialog
        from tkinter import *
        root = Tk()
        root.withdraw()
        curdir = f"{os.path.split(os.path.realpath(__file__))[0]}"
        folder_selected = filedialog.askdirectory(parent=root,
                                  initialdir=curdir,
                                  title='Select directory with .nd2 Files')
        if folder_selected != '':
            directory = folder_selected
    
    calc_proj = True
    create_fig = True
    compile_all = True

    export_flags = {
                    'raw':False,
                    'proj':True,
                    'figure':True
                    }

    channel_to_proj_map = {
                            '405':'mean',
                            '488':'mean',
                            '561':'mean',
                            '640':'mean'
                            }

    main(directory,clear=args.clear,groupby=(0,2),identify=(0,6),disp=args.show)
    



