## -*- coding: utf-8 -*-
"""ND2Figures
Version 1
Dillon Brown, 19Aug2021
"""
from plotting import *
from nd2merger import *
from findfiles import *

import os,argparse,json
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
    meth_map = {'mean':lambda x: np.nanmean(x[0:,:,:],axis=0,dtype=np.int32),
            'med':lambda x: np.nanmedian(x[0:,:,:],axis=0),
            'max':lambda x: np.nanmax(x[0:,:,:],axis=0),
            'sum':lambda x: np.nansum(x[0:,:,:],axis=0,dtype=np.int32),
    }
    img = np.int32(vol)
    proj = meth_map[method](img)
    proj = np.clip(proj, 0, 65535)
    return np.uint16(proj)

def find_frame_maxes(frames = []):
    norm_against = [np.max(x) for x in frames]
    overall_max = np.max(norm_against)
    norm_against = [np.divide(x,overall_max) for x in norm_against]
    norm_against = [int(np.round(x*255)) for x in norm_against]
    return norm_against

def plot_result(image, fg, bg):
    fg = np.nan_to_num(fg)
    bg = np.nan_to_num(bg)
    immax,bgmax = find_frame_maxes([image,bg])
    fig, ax = plt.subplots(nrows=1, ncols=3)
    bg_frame = normalize_frame(bg,bgmax)
    fg_frame = normalize_frame(fg,immax)
    image_frame = normalize_frame(image,immax)

    ax[0].imshow(bg_frame, cmap='Greens',vmin=0,vmax=immax)
    ax[0].set_title('Background')
    ax[0].axis('off')

    ax[1].imshow(fg_frame, cmap='Greens',vmin=0,vmax=immax)
    ax[1].set_title('Foreground')
    ax[1].axis('off')


    ax[2].imshow(image_frame, cmap='Greens',vmin=0,vmax=immax)
    ax[2].set_title('Original')
    ax[2].axis('off')

    fig.tight_layout()
    return fig

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

def write_proj(proj,path,filename,ext='.tif'):
    cv2.imwrite(f"{path}{os.sep}{filename}{ext}", proj)

def get_max(data,saturated:int=0.2):
    perc = 100-saturated
    result = np.percentile(np.ravel(data),perc,interpolation='nearest')
    return result

def identify_foreground(img):
    blur = cv2.GaussianBlur(img,(7,7),0)
    max_val = get_max(blur,saturated=70)
    blur = np.clip(blur, 0, max_val)
    thresh = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    thresh = cv2.adaptiveThreshold(thresh,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,1)
    kernel = np.ones((5,5),np.uint8)    
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel,iterations=2)
    kernel = np.ones((10,10),np.uint8)    
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel,iterations=1)
    kernel = np.ones((2,2),np.uint8)    
    thresh = cv2.dilate(thresh,kernel,iterations = 5*int(max(img.shape)/512))
    return thresh

##Main data analysis pipeline
def analyze_fstack(vol,proj_type,offset=True):

    proj = get_projection(vol,proj_type)
    if not offset:
        return proj,()
    # frame = normalize_frame(proj)

    mask_fg = identify_foreground(proj)

    mask_bg = cv2.bitwise_not(mask_fg)
    mask16_fg = cv2.normalize(mask_fg, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    mask16_bg = cv2.normalize(mask_bg, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
    foreground = cv2.bitwise_and(proj, mask16_fg)
    foreground = np.array(foreground, dtype=float)
    foreground[foreground==0] = np.nan
    background = cv2.bitwise_and(proj, mask16_bg)
    background = np.array(background, dtype=float)
    background[background==0] = np.nan

    mean_background = np.nanmean(background)

    offset = int(round(mean_background*1.5))
    if proj_type=='max':
        offset*=2
    offset_proj  = offset_projection(proj,offset)

    return offset_proj, (foreground,background), (mask_fg,mask_bg)



    
def main(directory,clear=True,groupby=None,identify=None,disp=False,offset=True):
    if disp:
        mpl.use("Qt5Agg")
    else:
        mpl.use("Agg")
    import matplotlib.pyplot as plt
    

    foldersWithData = scan_data(directory)

    ##Clear previous results
    if compile_all:
        raw_path,proj_path, fig_path,offset_path = create_folders(directory,export_flags,clear=clear)
    elif clear:
        [create_folders(folder,export_flags,clear=True) for folder in foldersWithData]

    plotter = SamplePlotter(norm_across_samples=True,norm_across_wavelengths = False)
    folder_count = 0
    samples = {}
    channel_maxes = {}
    image_outcomes_comp = {}

    for folder in foldersWithData:
        folder_count+=1

        if not compile_all:
            raw_path,proj_path, fig_path,offset_path = create_folders(folder,export_flags)
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
            chans = []
            image_outcomes = {}
            for i,(c,fstack) in enumerate(named_channels):
                projection_type = channel_to_proj_map[c]
                channel_outname = f'{sample.name}-{c}nm'
                projection_outname = f"{channel_outname}-{projection_type.upper()}"
                if export_flags['raw']:
                    imageio.mimwrite(f'{raw_path}{os.sep}{channel_outname}.tiff',fstack)
                vol = framestack_to_vol(fstack)
                ##Point at which custom analysis can be performed
                ##Function should take a 3d np.vol of uint16 and return a 2d array of uint16 for display
                ##
                proj, (fg, bg), (mask_fg,mask_bg) = analyze_fstack(vol,projection_type,offset=offset)
                image_outcomes[c] = {'fg':np.nanmean(fg),'bg':np.nanmean(bg)}
                top_percentile = get_max(proj)
                try:
                    if channel_maxes[c]<top_percentile:
                        channel_maxes[c] = top_percentile
                except:
                    channel_maxes[c] = top_percentile
                if offset&export_flags['offset']:
                    fig_fgbg = plot_result(proj,fg,bg)
                    fig_fgbg.savefig(f"{plotter.offset_out}-FGBG-{c}.png")
                    plt.close(fig_fgbg)
                write_proj(proj,proj_path,projection_outname)
                # write_proj(mask_fg,proj_path,f"{projection_outname}-fg_mask")
                
                chans.append((c,f"{proj_path}{os.sep}{projection_outname}.tif"))
            image_outcomes_comp[sample.name] = image_outcomes
            samples[sample] = (fig_path,chans)
    with open('analyzed.json','w') as fp:
        json.dump(image_outcomes_comp,fp)
        
    if create_fig:
        plotter.channel_maxes = channel_maxes
        for sample,(fig_path,channels) in samples.items():
            plotter.fig_folder = fig_path
            plotter.sample = sample
            plotter.init_plot()
            for c,projpath in channels:
                proj = cv2.imread(projpath,cv2.IMREAD_ANYDEPTH)
                plotter.set_channel(c,proj)
            plotter.plot()
            plotter.plot_composite()
            if disp:
                plt.show()
            if export_flags['figure']:
                plotter.save_fig()

            plt.close('all')
            print(samples)
if __name__=='__main__':
    curdir = f"{os.path.split(os.path.realpath(__file__))[0]}"
    previousFolderPath = f"{curdir}{os.sep}cache"
    LASTDIR = f"{previousFolderPath}{os.sep}LASTDIRECTORY.txt"

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
    parser.add_argument('--repeat', help="use same path as previous",
                                    default=False,
                                    action='store_true')
    parser.add_argument('--nooffset', help="Don't offset by auto-calculated background intensity",
                                    default=False,
                                    action='store_true')                                
    args = parser.parse_args()                     
    
    desktop = rf"D:"
    laptop = rf"C:\Users\dillo"
    ihc_dir = rf"{laptop}{os.sep}OneDrive - Georgia Institute of Technology\Thesis\Data\IHC"
    directory = "{ihc_dir}{os.sep}Confocal\20210827"
    if args.repeat:
        try:
            with open(LASTDIR, mode='r') as f:
                directory = json.load(f)
        except:
            pass
    elif not args.default:
        from tkinter import filedialog
        from tkinter import *
        root = Tk()
        root.withdraw()
        folder_selected = filedialog.askdirectory(parent=root,
                                initialdir=ihc_dir,
                                title='Select directory with .nd2 Files')
        if folder_selected != '':
            directory = folder_selected
    
    calc_proj = True
    create_fig = True
    compile_all = True

    export_flags = {
                    'raw':True,
                    'proj':True,
                    'figure':True,
                    'offset':True
                    }

    channel_to_proj_map = {
                            '405':'max',
                            '488':'mean',
                            '561':'mean',
                            '640':'mean'
                            }

    with open(LASTDIR, mode='w') as f:
        json.dump(directory,f)
    main(directory,clear=args.clear,groupby=(0,11),identify=(0,11),disp=args.show,offset=not args.nooffset)