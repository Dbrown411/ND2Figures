## -*- coding: utf-8 -*-
"""ND2Figures
Version 1
Dillon Brown, 19Aug2021
"""
import glob,os,argparse, json
import re as r
import itertools as it
from nd2reader import ND2Reader
import imageio
import wx
import numpy as np
import cv2 as cv2
from skimage import restoration

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.backend_bases import FigureCanvasBase as FigureCanvas
from matplotlib import transforms
from matplotlib_scalebar.scalebar import ScaleBar

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

app = wx.App(False) # the wx.App object must be created first.    
ppi = wx.Display().GetPPI()[0]
dispPPI = ppi*0.75


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

def set_axes_to_iterate(images):
    iteraxes = []
    if int(images.metadata['total_images_per_channel'])>1:
        iteraxes.append('z')
    if len(images.metadata['channels'])>1:
        iteraxes.append('c')
    images.iter_axes = iteraxes

def merge_files(collected_channels_dicts):
    collected_channels_dicts.sort(key=lambda x: len(x[1]))
    merged = collected_channels_dicts.pop(0)
    meta1,merged = merged
    collected_meta = [meta1]
    for m,d in collected_channels_dicts:
        merged.update(d)
        collected_meta.append([m])
    return (collected_meta, merged)

def resolve_channels(channels):
    cmap = {"640":'r',
            '561':'r',
            '488':'g',
            '405':'b'}
    if len(channels)>3:
        raise ValueError('More then 3 color channels not supported for creating composite images')
    if ('561' in channels)&('640' in channels):
        cmap = {
                "640":'r',
                '561':'g',
                '488':'b',
                '405':'b'}
    return cmap

def get_exportnames_from_file(pattern,identify,groupby):
    fig_ext = '.png'
    identifier_slice = identify if identify is not None else groupby if groupby is not None else (0,-1)
    sample_id = pattern[identifier_slice[0]:identifier_slice[1]]
    outname = "_".join(sample_id)
    fig_localoutname = f'{outname}{fig_ext}'
    return outname, fig_localoutname

def merge_nd2_to_dict(grouped_images,identify,groupby):
    collected_channels_dicts = []
    for i,(_,pattern,f) in enumerate(grouped_images):
        print(i,pattern,f)
        if i==0:
            outname, fig_localoutname = get_exportnames_from_file(pattern,identify,groupby)
            metadata = (outname,fig_localoutname)

        with ND2Reader(f) as images:
            channels = images.metadata['channels']
            numChannels = len(channels)
            set_axes_to_iterate(images)
            channels_dict = {channels[i]:[] for i in range(numChannels)}
            for k,frame in enumerate(images):
                channels_dict[channels[k%numChannels]].append(frame)
        collected_channels_dicts.append((images.metadata, channels_dict))
    tagged_collection = {'metadata':metadata,'data':collected_channels_dicts}
    
    return tagged_collection

def _check_all_folders_in_path(path):
    path_to_config = f"{path}{os.sep}channelmap.txt"
    if not os.path.isfile(path_to_config):
        try:
            _check_all_folders_in_path(os.path.split(path_to_config)[0])
        except:
            return None
    else:
        return path_to_config

def get_channelmap(folder):
    print(folder)
    path_to_config = _check_all_folders_in_path(folder)
    if path_to_config is None:
        with open('default_channelmap.txt','r') as f:
            channel_to_protein = json.load(f)
    else:
        with open(path_to_config,'r') as f:
            channel_to_protein = json.load(f)

    print(channel_to_protein)
    return channel_to_protein

##Data functions (operating on u16)
normalize_frame = lambda x: cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def framestack_to_vol(framestack):
    volume_shape = (len(framestack),framestack[0].shape[0],framestack[0].shape[1])
    vol = np.empty(volume_shape,dtype=np.uint16)
    for i,frame in enumerate(framestack):
        vol[i,:,:] = frame
    return vol

def get_projection(framestack,method='mean'):
    meth_map = {'mean':lambda x: np.nanmean(x[0:,:,:],axis=0,dtype=np.uint16),
            'med':lambda x: np.nanmedian(x[0:,:,:],axis=0),
            'max':lambda x: np.nanmax(x[0:,:,:],axis=0),
            'sum':lambda x: np.nansum(x[0:,:,:],axis=0,dtype=np.uint16)
    }
    vol = framestack_to_vol(framestack)
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


##Plotting Functions

##Data (u16) to Frame (u8 RGB)

def gray_to_color(frame,color='g'):
    cmap = {"r":-2,'g':-1,'b':0}
    dst = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    dst[:,:,:2] = 0
    dst = np.roll(dst,cmap[color],axis=2)
    return dst

def projection_to_frame(proj,c):
    frame = normalize_frame(proj)
    color_frame = gray_to_color(frame,c)
    return color_frame

##Frame functions for display
def make_composite(cframes):
    shapes = [x.shape for x in cframes]
    if len(set(shapes)) > 1:
        return None
    result = np.nansum(np.stack(cframes),axis=0)
    return result

def show_frame(frame,ax):
    interp='hermite'
    if frame is None:
        return
    ax.imshow(frame,interpolation=interp,vmin=0, vmax=255, aspect='equal') 

def calc_fontsize(frameshape):
    subplot_size =(frameshape[1]/ppi,frameshape[0]/ppi)
    normalizing_dim = subplot_size[1]*72
    fontsize = int(round(normalizing_dim*0.06))
    print(f"ScanDimensions: {frameshape[1]}x{frameshape[0]}px\nFigDimensions: {subplot_size[0]:.02f}x{subplot_size[1]:.02f}in.\nFontsize: {fontsize}pt")
    return fontsize

def rainbow_text(x,y,ls,lc,ax,**kw):
    t = ax.transAxes
    fig = plt.gcf()
    for i,b in enumerate(zip(ls,lc)):
        s,c = b
        text = plt.text(x,y,s,color=c, transform=t,**kw)
        text.draw(fig.canvas.get_renderer())
        ex = text.get_window_extent()
        t = transforms.offset_copy(text._transform, x=ex.width, units='dots')

def set_plot(framesize,n):
    if framesize[1]>framesize[0]:
        ncols = 1
        nrows = n
        figsize = (framesize[1]/ppi,(framesize[0]*nrows/ppi)*(1/.9))
    else:
        ncols = n
        nrows = 1
        figsize = (framesize[1]*ncols/ppi,(framesize[0]/ppi)*(1/.9))

    fig = plt.figure(dpi=ppi,figsize=figsize)
    gs = mpl.pyplot.GridSpec(ncols=ncols, nrows=nrows,wspace=0.01,hspace=0.01)
    l, t, r, b = 0, .9, 1, 0 
    gs.update(left=l, top=t, right=r, bottom=b)
    return fig,gs

def yield_subplot(col,fig,gs):
    try:
        ax1 = fig.add_subplot(gs[0,col])
    except:
        ax1 = fig.add_subplot(gs[col,0])
        
    ax1.axis('off')
    return ax1


##IO
def write_proj(fig,path,filename,ext='.tif'):
    cv2.imwrite(f"{path}{os.sep}{filename}{ext}", fig)

def save_fig(fig,f):
    try:
        fig.savefig(f)
    except:
        try:
            fig.savefig(f'{f}-01')
        except Exception as e:
            print(e)
            pass

 
def main(directory,clear=True,groupby=None,identify=None):

    foldersWithData = scan_data(directory)
    [print(folder) for folder in foldersWithData]

    ##Clear previous results
    if compile_all:
        raw_path,proj_path, fig_path = create_folders(directory,export_flags,clear=clear)
    elif clear:
        [create_folders(folder,export_flags,clear=True) for folder in foldersWithData]

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

        sample_count = 0
        for group,grouped_images in file_groups:
            sample_count+=1

            tagged_data = merge_nd2_to_dict(grouped_images,identify,groupby)

            ##get chosen sample identifer from tagged data
            outname,fig_localoutname = tagged_data.pop('metadata')
            fig_outname = f'{fig_path}{os.sep}{fig_localoutname}'
            print(f'\nSample {sample_count}: {outname}\n')
            
            ##get collection of data and merge
            collected_channels_dicts = tagged_data['data']
            metadata, channels_dict = merge_files(collected_channels_dicts)
            c,first_channel = next(iter((channels_dict.items())))

            channels = list(channels_dict.keys())
            channels.sort(reverse=True)
            named_channels = list(channels_dict.items())
            named_channels.sort(key=lambda x: x[0],reverse=True)
            channel_to_color = resolve_channels(channels)
            channel_to_protein = get_channelmap(folder)

            print('Recognized Channels:')
            [print(f"{x}nm") for x in channels]
            print('')

            
            ##Data calculations
            
            first_frame = first_channel[0]

            ##Prepare plot
            num_subplots = len(channels)
            if num_subplots>1:
                num_subplots+=1
            framesize = first_frame.shape
            fig,gs = set_plot(framesize,num_subplots)
            fontsize = calc_fontsize(framesize)
            fig.suptitle(outname,fontsize=fontsize)
            

            color_frames = []

            for i,(c,fstack) in enumerate(named_channels):
                projection_type = channel_to_proj_map[c]
                channel_outname = f'{outname}-{c}nm-{projection_type.upper()}'
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
                    color_frame = projection_to_frame(proj,channel_to_color[c])
                    ax = yield_subplot(i,fig,gs)
                    show_frame(color_frame, ax)

                    try:
                        pixels_micron = metadata['pixel_microns']
                    except:
                        pixels_micron = metadata[0]['pixel_microns']

                    if i==0:
                        scalebar = ScaleBar(pixels_micron,'um',frameon=True,location='lower right',
                                            box_color=(1, 1, 1),box_alpha = 0,color='white',
                                            font_properties = {'size':int(round(fontsize/2))})
                        ax.add_artist(scalebar)
                if create_fig:
                    rainbow_text(0.03,.02,[channel_to_protein[c]],[channel_to_color[c]],size=fontsize,ax=ax)
                    color_frames.append(color_frame)

            if (len(color_frames)>1)&(create_fig):
                ax = yield_subplot(num_subplots-1,fig,gs)
                composite = make_composite(color_frames)
                if composite is None:
                    continue
                show_frame(composite, ax)
                rainbow_text(0.03,.02,[channel_to_protein[x] for x in channels],[channel_to_color[x] for x in channels],size=fontsize,ax=ax)
            if create_fig:
                plt.show()
                if export_flags['figure']:
                    save_fig(fig,fig_outname)

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

    if args.show:
        import matplotlib as mpl
        mpl.use("Qt5Agg")
    else:
        import matplotlib as mpl
        mpl.use("Agg")
    import matplotlib.pyplot as plt

    
    laptop = r'C:\Users\dillo'
    desktop = r'D:'
    directory = rf"{laptop}{os.sep}OneDrive - Georgia Institute of Technology\Lab\Data\IHC\Confocal\Automated"
    if not args.default:
        from tkinter import filedialog
        from tkinter import *
        root = Tk()
        root.withdraw()
        curdir = f"{os.path.split(os.path.realpath(__file__))[0]}"
        print(curdir)
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

    channel_to_protein = {
                        '405':'DAPI',
                        '488':'ACAN',
                        '561':'ACAN',
                        '640':'pACAN'
                        }


    channel_to_proj_map = {
                            '405':'max',
                            '488':'mean',
                            '561':'mean',
                            '640':'mean'
                            }

    main(directory,clear=args.clear,groupby=(0,2),identify=(0,6))
    



