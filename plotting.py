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
    def __init__(self,nd2images,disp=False):
        app = wx.App(False) # the wx.App object must be created first.    
        self.ppi = wx.Display().GetPPI()[0]
        self.dispPPI = self.ppi*0.75
        self.nd2images = nd2images
        self.set_plotvisible(disp)
        self.fig_ext = '.png'

    def set_plotvisible(self,disp):
        self.mpl = import_module('matplotlib')
        if disp:
            self.mpl.use("Qt5Agg")
        else:
            self.mpl.use("Agg")
        self.plt = import_module('matplotlib.pyplot')
        self.GridSpec = import_module('matplotlib.pyplot.GridSpec')    
    
    @property
    def nd2images(self):
        return self._nd2images
    @nd2images.setter
    def nd2images(self,nd2images):
        self._nd2images = nd2images
        if nd2images is None:
            return
        
        self.outname = nd2images.outname
        self.fig_localout = f'{self.outname}{self.fig_ext}'
        num_subplots = len(nd2images.channels)
        if num_subplots>1:
            num_subplots+=1
        self.num_subplots = num_subplots
        self.framesize = nd2images.framesize
        self.set_plot()
    
    ##Plotting Functions
    ##Data (u16) to Frame (u8 RGB)

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
    
    def rainbow_text(self,x,y,ls,lc,ax,**kw):
        t = ax.transAxes
        fig = self.plt.gcf()
        for i,b in enumerate(zip(ls,lc)):
            s,c = b
            text = self.plt.text(x,y,s,color=c, transform=t,**kw)
            text.draw(fig.canvas.get_renderer())
            ex = text.get_window_extent()
            t = transforms.offset_copy(text._transform, x=ex.width, units='dots')

    def _calc_fontsize(self):
        frameshape = self.framesize
        subplot_size =(frameshape[1]/self.ppi,frameshape[0]/self.ppi)
        normalizing_dim = subplot_size[1]*72
        self.fontsize = int(round(normalizing_dim*0.06))
        print(f"ScanDimensions: {frameshape[1]}x{frameshape[0]}px\nFigDimensions: {subplot_size[0]:.02f}x{subplot_size[1]:.02f}in.\nFontsize: {self.fontsize}pt")

    def _calc_figsize(self):
        framesize = self.framesize
        n = self.num_subplots

        if framesize[1]>framesize[0]:
            ncols = 1
            nrows = n
            self.figsize = (framesize[1]/self.ppi,(framesize[0]*nrows/self.ppi)*(1/.9))
        else:
            ncols = n
            nrows = 1
            self.figsize = (framesize[1]*ncols/self.ppi,(framesize[0]/self.ppi)*(1/.9))
        self.plotshape = (ncols,nrows)

    def set_plot(self):
        self._calc_figsize()
        self._calc_fontsize()
        ncols,nrows = self.plotshape
        self.fig = self.plt.figure(dpi=self.ppi,figsize=self.figsize)
        self.gs = self.GridSpec(ncols=ncols, nrows=nrows,wspace=0.01,hspace=0.01)
        l, t, r, b = 0, .9, 1, 0 
        self.gs.update(left=l, top=t, right=r, bottom=b)

    def yield_subplot(self,col):
        try:
            ax1 = self.fig.add_subplot(self.gs[0,col])
        except:
            ax1 = self.fig.add_subplot(self.gs[col,0])
            
        ax1.axis('off')
        return ax1

    def save_fig(f):

        try:
            self.fig.savefig(f)
        except:
            try:
                fig.savefig(f'{f}-01')
            except Exception as e:
                print(e)
                pass


