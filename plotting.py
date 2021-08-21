import cv2 as cv2
import numpy as np
import wx, json, os
from matplotlib import transforms
from matplotlib.pyplot import GridSpec
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

normalize_frame = lambda x: cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
class SamplePlotter:
    def __init__(self):
        app = wx.App(False) # the wx.App object must be created first.    
        self.ppi = wx.Display().GetPPI()[0]
        self.dispPPI = self.ppi*0.75
        self.fig_ext = '.tif'
    @property
    def sample(self):
        return self._sample
    @sample.setter
    def sample(self,sample):
        self._sample = sample
        self.color_frames = []
        self.color_data = []
        if sample is None:
            return
        
        self.sample_name = sample.name
        self.figname = f'{self.sample_name}{self.fig_ext}'
        self.fig_out = f"{self.fig_folder}{os.sep}{self.figname}"
        num_subplots = len(sample.channels)
        self.channels = sample.channels
        if num_subplots>1:
            num_subplots+=1
        self.num_subplots = num_subplots
        self.framesize = sample.framesize
        self.set_plot()
        self.make_scalebar(self.sample.metadata)

    @property
    def fig_folder(self):
        return self._fig_folder
    @fig_folder.setter
    def fig_folder(self,fig_folder):
        self._fig_folder = fig_folder



    ##Plotting Functions
    def make_scalebar(self,metadata):
        try:
            pixels_micron = metadata['pixel_microns']
        except Exception as e:
            pixels_micron = metadata[0]['pixel_microns']
        self.scalebar = ScaleBar(pixels_micron,'um',frameon=True,location='lower right',
                            box_color=(1, 1, 1),box_alpha = 0,color='white',
                            font_properties = {'size':int(round(self.fontsize/2))})

    ##Data (u16) to Frame (u8 RGB)
    def set_channel(self,c,data):
        color_16 = self.gray_to_color(data,self.sample.channel_to_color[c])
        self.color_data.append(color_16)

    def plot_channel(self,c,normalize=True,norm_max = 255):
        
        subplot_num = self.sample.channel_order[c]
        color_frame = self.color_data[subplot_num]
        if normalize:
            color_frame = cv2.normalize(color_frame, None, 0, norm_max, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        ax = self.yield_subplot(subplot_num)
        if subplot_num==0:
            self.ax0 = ax
        self.show_frame(color_frame, ax)
        ls = [self.sample.channel_to_protein[c]]
        lc = [self.sample.channel_to_color[c]]
        self.rainbow_text(0.03,.02,ls,lc,size=self.fontsize,ax=ax)
        self.color_frames.append(color_frame)

    def plot(self,normalize=('488','561','640')):
        channel_norm = {k:255 for k in self.sample.channel_order.keys()}
        
        indices = [(c,self.sample.channel_order[c]) for c in normalize if c in self.sample.channel_order.keys()]
        if len(indices)>1:
            normalizing = [(c,self.color_data[i]) for c,i in indices]
            norm_against = [(c,np.max(x)) for c,x in normalizing]
            c,maxes = zip(*norm_against)
            overall_max = np.max(maxes)
            norm_against = [(c,int(np.round((np.divide(x,overall_max)))*255)) for c,x in norm_against]
            for c,m in norm_against:
                channel_norm[c]=m

        for c,i in self.sample.channel_order.items():
            self.plot_channel(c,normalize=True,norm_max = channel_norm[c])

    def gray_to_color(self,frame,color='g'):
        cmap = {"r":-2,'g':-1,'b':0}
        dst = np.stack((frame,)*3,axis=-1)
        dst[:,:,:2] = 0
        dst = np.roll(dst,cmap[color],axis=2)
        return dst

    def projection_to_frame(self,proj,c):
        frame = normalize_frame(proj)
        color_frame = self.gray_to_color(frame,c)
        return color_frame

    ##Frame functions for display
    def plot_composite(self):
        self.ax0.add_artist(self.scalebar)
        if len(self.color_frames)>3:
            print('too many channels')
            return None
        composite = self.make_composite()

        ax = self.yield_subplot(self.num_subplots-1)
        self.show_frame(composite, ax)
        ls= [self.sample.channel_to_protein[x] for x in self.channels]
        lc=[self.sample.channel_to_color[x] for x in self.channels]
        self.rainbow_text(0.03,.02,ls,lc,size=self.fontsize,ax=ax)


    def make_composite(self):
        shapes = [x.shape for x in self.color_frames]
        if len(set(shapes)) > 1:
            return None
        result = np.nansum(np.stack(self.color_frames),axis=0)
        return result

    def show_frame(self,frame,ax):
        interp='hermite'
        if frame is None:
            return
        ax.imshow(frame,interpolation=interp,vmin=0, vmax=255, aspect='equal') 

    
    def rainbow_text(self,x,y,ls,lc,ax,**kw):
        t = ax.transAxes
        fig = plt.gcf()

        for i,b in enumerate(zip(ls,lc)):
            s,c = b
            text = plt.text(x,y,s,color=c, transform=t, **kw)
            text.set_path_effects([PathEffects.withStroke(linewidth=4, foreground='k')])

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
        self.fig = plt.figure(dpi=self.ppi,figsize=self.figsize)
        self.gs = GridSpec(ncols=ncols, nrows=nrows,wspace=0.01,hspace=0.01)
        l, t, r, b = 0, .9, 1, 0 
        self.gs.update(left=l, top=t, right=r, bottom=b)
        self.fig.suptitle(self.figname[:-4],fontsize=self.fontsize)

    
    def yield_subplot(self,col):
        try:
            ax1 = self.fig.add_subplot(self.gs[0,col])
        except:
            ax1 = self.fig.add_subplot(self.gs[col,0])
            
        ax1.axis('off')
        return ax1

    def save_fig(self):
        try:
            self.fig.savefig(self.fig_out)
        except:
            try:
                self.fig.savefig(f'{self.fig_out}-01')
            except Exception as e:
                print(e)
                pass


