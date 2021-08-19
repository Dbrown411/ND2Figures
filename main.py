from nd2reader import ND2Reader
import matplotlib as mpl
mpl.use("Qt5Agg")
import matplotlib.pyplot as plt
import glob,os
import numpy as np
import cv2 as cv2
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.backend_bases import FigureCanvasBase as FigureCanvas
import wx
from matplotlib_scalebar.scalebar import ScaleBar
import re as r
from matplotlib import transforms
import itertools as it
"""
Recursively crawl parent directory,
find all .nd2 file types,
groupby (ID1,ID2) from file name using naming convention

    "ID1{-|_|.}ID2{-|_|.}ADDITIONAL.nd2"

"""



app = wx.App(False) # the wx.App object must be created first.    
ppi = wx.Display().GetPPI()[0]
dispPPI = ppi*0.75


def rainbow_text(x,y,ls,lc,ax,**kw):
    t = ax.transAxes
    fig = plt.gcf()
    # plt.show()

    #horizontal version
    for i,b in enumerate(zip(ls,lc)):
        print(i)
        s,c = b
        print(s)
        if i>0:
            x=0.08
        text = plt.text(x,y,' '+s,color=c, transform=t, ha='left', va='bottom',**kw)
        text.draw(fig.canvas.get_renderer())
        ex = text.get_window_extent()
        t = transforms.offset_copy(text._transform, x=ex.width, units='dots')

    # #vertical version
    # for s,c in zip(ls,lc):
    #     text = plt.text(x,y," "+s+" ",color=c, transform=t,
    #             rotation=90,va='bottom',ha='center',**kw)
    #     text.draw(fig.canvas.get_renderer())
    #     ex = text.get_window_extent()
    #     t = transforms.offset_copy(text._transform, y=ex.height, units='dots')


def framestack_to_vol(framestack):
    volume_shape = (len(framestack),framestack[0].shape[0],framestack[0].shape[1])
    vol = np.empty(volume_shape,dtype=np.uint16)
    for i,frame in enumerate(framestack):
        vol[i,:,:] = frame
    return vol

def get_projection(framestack,method='mean'):
    meth_map = {'mean':lambda x: np.nanmean(x[0:,:,:],axis=0,dtype=np.uint16),
            'med':lambda x: np.nanmedian(x[0:,:,:],axis=0),
            'max':lambda x: np.nanmax(x[0:,:,:],axis=0,dtype=np.uint16),
            'sum':lambda x: np.nansum(x[0:,:,:],axis=0,dtype=np.uint16)
    }
    vol = framestack_to_vol(framestack)
    VIP = meth_map[method](vol)
    return VIP

def write_proj(fig,path,filename,ext='.tif'):
    cv2.imwrite(f"{path}{os.sep}{filename}{ext}", fig)

def save_fig(fig,f):
    try:
        fig.savefig(f, dpi=ppi)
    except:
        try:
            fig.savefig(f'{f}-01', dpi=ppi)
        except Exception as e:
            print(e)
            pass

def gray_to_color(frame,color='g'):
    cmap = {"640":-2,'488':-1,'405':0}
    dst = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    dst[:,:,:2] = 0
    dst = np.roll(dst,cmap[color],axis=2)
    return dst

def set_plot(figsize,ncols):
    fig = plt.figure(dpi=dispPPI,figsize=(figsize[1]*ncols/ppi,(figsize[0]/ppi)*(1/.9)))
    gs = mpl.pyplot.GridSpec(ncols=ncols, nrows=1,wspace=0.02)
    l, t, r, b = 0, .9, 1, 0 
    gs.update(left=l, top=t, right=r, bottom=b)
    return fig,gs

def yield_subplot(col,fig,gs):
    ax1 = fig.add_subplot(gs[0,col])
    ax1.axis('off')
    return ax1

def show_frame(frame,ax):
    interp='hermite'
    if frame is None:
        return
    ax.imshow(frame,interpolation=interp,vmin=0, vmax=255, aspect='equal') 

def create_folders(folder):
    projection_path = f"{folder}{os.sep}16bitProjections"
    figure_path = f"{folder}{os.sep}Figures"
    try:
        os.mkdir(projection_path)
    except Exception as e:
        pass
    try:
        os.mkdir(figure_path)
    except Exception as e:
        pass
    return projection_path, figure_path

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

def match_scans(folder):
    nd2Files = list(glob.glob(folder+os.sep+'*.nd2'))
    nd2Filenames = [(i,r.split(r'-|_|\.',os.path.split(file)[1].replace(' ','').upper()),file) for i, file in enumerate(nd2Files)]
    grouped = it.groupby(nd2Filenames,key=lambda x: x[1][:2])
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


def make_composite(cframes):
    shapes = [x.shape for x in cframes]
    if len(set(shapes)) > 1:
        return None
    result = np.nansum(np.stack(cframes),axis=0)
    return result


stainMap = {'405':'DAPI',
            '488':'ACAN',
            '640':'pACAN'}
wavelength_to_color = {"640":'r','488':'g','405':'b'}



if __name__=='__main__':
    laptop = r'C:\Users\dillo'
    desktop = r'D:'
    directory = rf"{desktop}{os.sep}OneDrive - Georgia Institute of Technology\Lab\Data\IHC\Confocal\Automated"
    normalize_frame = lambda x: cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    foldersWithData = scan_data(directory)
    for folder in foldersWithData:
        proj_path, fig_path = create_folders(folder)
        file_groups = match_scans(folder)
        fig_ext = '.png'
        fontsize = 50
        print('--'*10)
        print(folder)
        for sample_id,grouped_images in file_groups:
            print(sample_id)
            outname = "_".join(sample_id)
            fig_outname = f'{fig_path}{os.sep}{"_".join(sample_id)}{fig_ext}'
            
            collected_channels_dicts = []
            for i,pattern,f in grouped_images:
                path,filename = os.path.split(f)
                outname= filename[:-4]
                with ND2Reader(f) as images:
                    channels = images.metadata['channels']
                    numChannels = len(channels)
                    set_axes_to_iterate(images)
                    channels_dict = {channels[i]:[] for i in range(numChannels)}
                    for i,frame in enumerate(images):
                        channels_dict[channels[i%numChannels]].append(frame)
                collected_channels_dicts.append((images.metadata, channels_dict))
            
            metadata, channels_dict = merge_files(collected_channels_dicts)
            num_subplots = len(channels_dict.keys())
            if num_subplots>1:
                num_subplots+=1
            
            framesize = frame.shape
            fig,gs = set_plot(framesize,num_subplots)
            fig.suptitle(outname)
            color_frames = []
            named_channels = list(channels_dict.items())
            channels = []
            named_channels.sort(key=lambda x: x[0],reverse=True)
            for i,(c,fstack) in enumerate(named_channels):
                channels.append(c)
                ax = yield_subplot(i,fig,gs)
                proj = get_projection(fstack)
                try:
                    pixels_micron = metadata['pixel_microns']
                except:
                    pixels_micron = metadata[0]['pixel_microns']
                scalebar = ScaleBar(pixels_micron,'um',frameon=True,location='lower right',box_color=(1, 1, 1),box_alpha = 0,color='white',font_properties = {'size':fontsize/2})

                frame = normalize_frame(proj)
                color_frame = gray_to_color(frame,c)
                show_frame(color_frame, ax)
                if i==0:
                    ax.add_artist(scalebar)
                ax.text(0.03,.02,stainMap[c],ha='left',va='bottom',color=wavelength_to_color[c],transform=ax.transAxes,size=fontsize)
                write_proj(proj,proj_path,f'{outname}-{c}nm')
                color_frames.append(color_frame)
            if len(color_frames)>1:
                ax = yield_subplot(num_subplots-1,fig,gs)
                composite = make_composite(color_frames)
                if composite is None:
                    continue
                show_frame(composite, ax)
                rainbow_text(0.03,.02,[stainMap[x] for x in channels],[wavelength_to_color[x] for x in channels],size=fontsize,ax=ax)
            # plt.show()
            save_fig(fig,fig_outname)
            plt.close()



