import cv2 as cv2
import numpy as np
from matplotlib import transforms
import wx, json, os
from importlib import import_module


normalize_frame = lambda x: cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def _check_all_folders_in_path(path):
    path_to_config = f"{path}{os.sep}channelmap.txt"
    if not os.path.isfile(path_to_config):
        try:
            _check_all_folders_in_path(os.path.split(path_to_config)[0])
        except:
            return None
    else:
        return path_to_config

class ProjPlotter:
    def __init__(self,disp=False):
        app = wx.App(False) # the wx.App object must be created first.    
        self.ppi = wx.Display().GetPPI()[0]
        self.dispPPI = self.ppi*0.75
        self._proj = None
        self.set_plotvisible(disp)
        self._set_defaults()

    @property
    def folder(self):
        return self._folder
    @folder.setter
    def folder(self,folder):
        self._folder = folder
        self.get_channelmap(folder)
    
    @property
    def channels(self):
        return self._channels
    @channels.setter
    def channels(self,channels):
        self._channels = channels
        self.resolve_channels()
    
    @property
    def proj(self):
        return self._proj
    @proj.setter
    def proj(self,proj):
        self._proj = proj
        

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

    def get_channelmap(self,folder):
        path_to_config = _check_all_folders_in_path(folder)
        if path_to_config is None:
            try:
                with open('default_channelmap.txt','r') as f:
                    self.channel_to_protein = json.load(f)
            except:
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

    ##Plotting Functions
    ##Data (u16) to Frame (u8 RGB)
    def set_plotvisible(self,disp):
        self.mpl = import_module('matplotlib')
        if disp:
            self.mpl.use("Qt5Agg")
        else:
            self.mpl.use("Agg")
        self.plt = import_module('matplotlib.pyplot')
        self.GridSpec = import_module('matplotlib.pyplot.GridSpec')
    def gray_to_color(self,frame,color='g'):
        cmap = {"r":-2,'g':-1,'b':0}
        dst = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        dst[:,:,:2] = 0
        dst = np.roll(dst,cmap[color],axis=2)
        return dst

    def projection_to_frame(self,proj,c):
        frame = normalize_frame(proj)
        color_frame = self.gray_to_color(frame,c)
        return color_frame

    ##Frame functions for display
    def make_composite(self,cframes):
        shapes = [x.shape for x in cframes]
        if len(set(shapes)) > 1:
            return None
        result = np.nansum(np.stack(cframes),axis=0)
        return result

    def show_frame(self,frame,ax):
        interp='hermite'
        if frame is None:
            return
        ax.imshow(frame,interpolation=interp,vmin=0, vmax=255, aspect='equal') 

    def calc_fontsize(self,frameshape):
        subplot_size =(frameshape[1]/self.ppi,frameshape[0]/self.ppi)
        normalizing_dim = subplot_size[1]*72
        fontsize = int(round(normalizing_dim*0.06))
        print(f"ScanDimensions: {frameshape[1]}x{frameshape[0]}px\nFigDimensions: {subplot_size[0]:.02f}x{subplot_size[1]:.02f}in.\nFontsize: {fontsize}pt")
        return fontsize

    def rainbow_text(self,x,y,ls,lc,ax,**kw):
        t = ax.transAxes
        fig = self.plt.gcf()
        for i,b in enumerate(zip(ls,lc)):
            s,c = b
            text = self.plt.text(x,y,s,color=c, transform=t,**kw)
            text.draw(fig.canvas.get_renderer())
            ex = text.get_window_extent()
            t = transforms.offset_copy(text._transform, x=ex.width, units='dots')

    def set_plot(self,framesize,n):
        if framesize[1]>framesize[0]:
            ncols = 1
            nrows = n
            figsize = (framesize[1]/self.ppi,(framesize[0]*nrows/self.ppi)*(1/.9))
        else:
            ncols = n
            nrows = 1
            figsize = (framesize[1]*ncols/self.ppi,(framesize[0]/self.ppi)*(1/.9))

        fig = self.plt.figure(dpi=self.ppi,figsize=figsize)
        gs = self.GridSpec(ncols=ncols, nrows=nrows,wspace=0.01,hspace=0.01)
        l, t, r, b = 0, .9, 1, 0 
        gs.update(left=l, top=t, right=r, bottom=b)
        return fig,gs

    def yield_subplot(self,col,fig,gs):
        try:
            ax1 = fig.add_subplot(gs[0,col])
        except:
            ax1 = fig.add_subplot(gs[col,0])
            
        ax1.axis('off')
        return ax1

