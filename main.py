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

app = wx.App(False) # the wx.App object must be created first.    
ppi = wx.Display().GetPPI()[0]
dispPPI = ppi*0.75
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
    fig.savefig(f, dpi=ppi)

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
    nd2Filenames = [(i,r.split(r'-|_',os.path.split(file)[1].replace(' ','').upper())) for i, file in enumerate(nd2Files)]
    for ind,data in nd2Filenames:
        print(ind,data)

def set_axes_to_iterate(images):
    iteraxes = []
    if int(images.metadata['total_images_per_channel'])>1:
        iteraxes.append('z')
    if len(images.metadata['channels'])>1:
        iteraxes.append('c')
    images.iter_axes = iteraxes

if __name__=='__main__':
    directory = r"D:\OneDrive - Georgia Institute of Technology\Lab\Data\IHC\Confocal\Automated"
    normalize_frame = lambda x: cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    foldersWithData = scan_data(directory)
    # foldersWithData=[foldersWithData[0]]
    for folder in foldersWithData:
        proj_path, fig_path = create_folders(folder)
        # print(folder)
        # match_scans(folder)
        for f in glob.glob(folder+os.sep+'*.nd2'):
            path,filename = os.path.split(f)
            outname= filename[:-4]
            out_ext = '.tif'
            with ND2Reader(f) as images:
                channels = images.metadata['channels']
                numChannels = len(channels)
                set_axes_to_iterate(images)
                scalebar = ScaleBar(images.metadata['pixel_microns'],'um',frameon=True,location='lower right',box_color=(1, 1, 1),box_alpha = 0,color='white')
                channels_dict = {i:[] for i in range(numChannels)}
                for i,frame in enumerate(images):
                    channels_dict[i%numChannels].append(frame)
                    framesize = frame.shape

                num_subplots = len(channels_dict.keys())
                if num_subplots>1:
                    num_subplots+=1

                fig,gs = set_plot(framesize,num_subplots)
                fig.suptitle(filename[:-4])
                color_frames = []
                for k,fstack in channels_dict.items():
                    ax = yield_subplot(k,fig,gs)
                    proj = get_projection(fstack)

                    frame = normalize_frame(proj)
                    color_frame = gray_to_color(frame,channels[k])
                    show_frame(color_frame, ax)
                    if k==0:
                        ax.add_artist(scalebar)
                    write_proj(proj,proj_path,f'{outname}-{channels[k]}nm')
                    color_frames.append(color_frame)
                if len(color_frames)>1:
                    ax = yield_subplot(num_subplots-1,fig,gs)
                    show_frame(np.nansum(np.asarray(color_frames),axis=0), ax)
                #plt.show()
                figname = rf"{fig_path}{os.sep}{outname}-Compiled{out_ext}"
                save_fig(fig,figname)
                plt.close()



