from app.plotter import *
from app.accumulator import *
from app.utilities import *
from app.analysis import *
from app.config import ConfigContainer as cc

import imageio
import numpy as np
import cv2 as cv2
import matplotlib as mpl
from typing import Tuple
from pathlib import Path
from tqdm import tqdm
import tifffile


def initialize_folders(
    directory: Path,
    clear: bool,
) -> ExportLocation:
    export_paths = create_folders(directory, cc.export_flags, clear=clear)
    return export_paths


def nd2_to_tiff():
    pass


def analyze_folder(directory: Path,
                   clear: bool = True,
                   groupby: Tuple[int] = None,
                   identify: Tuple[int] = None,
                   disp: bool = False,
                   offset: bool = True):
    if disp:
        mpl.use("Qt5Agg")
    else:
        mpl.use("Agg")
    import matplotlib.pyplot as plt

    folders_with_data = scan_data(directory)
    export_locations = initialize_folders(directory, clear)
    plotter = SamplePlotter(norm_across_samples=True,
                            norm_across_wavelengths=False)
    samples = {}
    channel_maxes = {}
    image_outcomes_comp = {}
    unique_samples = []
    for folder in folders_with_data:
        unique_samples.extend(match_scans(folder, groupby=groupby))

    if cc.show_progress:
        pbar = tqdm(list(reversed(unique_samples)))
    else:
        pbar = reversed(unique_samples)
    for group in pbar:
        if cc.show_progress:
            pbar.set_description(group[-1][0].stem)

        sample = ND2Accumulator(group, identify, groupby)
        plotter.fig_folder = export_locations.fig
        plotter.offset_folder = export_locations.offset
        plotter.sample = sample
        named_channels = sample.named_channels
        chans = []
        image_outcomes = {}
        x, y = named_channels[0][1][0].shape
        z = len(named_channels[0][1])
        imgs = np.empty((z, x, y, 3))

        def channel_to_rgb(c: str):
            c = int(c)
            if c > 580:
                return 0
            if c < 480:
                return 2
            return 1

        for i, (c, fstack) in enumerate(named_channels):
            color = channel_to_rgb(c)
            imgs[:, :, :, color] = np.stack(fstack, axis=0)
        if z == 1:
            imgs = np.squeeze(imgs, axis=0)

        if cc.export_flags['raw']:
            tifffile.imsave(
                export_locations.raw / f'{sample.name}-composite.tiff', imgs)

        for i, (c, fstack) in enumerate(named_channels):
            projection_type = cc.channel_to_proj_map[c]
            channel_outname = f'{sample.name}-{c}nm'
            projection_outname = f"{channel_outname}-{projection_type.upper()}"
            # if cc.export_flags['raw']:
            #     imageio.mimwrite(
            #         export_locations.raw / f'{channel_outname}.tiff', fstack)
            vol = framestack_to_vol(fstack)
            ##Point at which custom analysis can be performed
            ##Function should take a 3d np.vol of uint16 and return a 2d array of uint16 for display
            ##
            proj, (fg, bg), (mask_fg, mask_bg) = analyze_fstack(
                vol,
                projection_type,
                offset=offset,
                **{'label': channel_outname})
            image_outcomes[c] = {'fg': np.nanmean(fg), 'bg': np.nanmean(bg)}
            top_percentile = get_max(proj)
            try:
                if channel_maxes[c] < top_percentile:
                    channel_maxes[c] = top_percentile
            except:
                channel_maxes[c] = top_percentile
            if offset & cc.export_flags['offset']:
                fig_fgbg = plot_result(proj, fg, bg)
                fig_fgbg.savefig(f"{plotter.offset_out}-FGBG-{c}.png")
                plt.close(fig_fgbg)
            write_proj(proj, export_locations.proj, projection_outname)
            export_path = export_locations.proj / f"{projection_outname}.tif"
            chans.append((c, export_path))
        image_outcomes_comp[sample.name] = image_outcomes
        samples[sample] = (export_locations.fig, chans)
    with open(export_locations.parent / 'analyzed.json', 'w') as fp:
        json.dump(image_outcomes_comp, fp)

    if cc.create_fig:
        plotter.channel_maxes = channel_maxes
        for sample, (export_locations.fig, channels) in samples.items():
            plotter.fig_folder = export_locations.fig
            plotter.sample = sample
            plotter.init_plot()
            for c, projpath in channels:
                proj = cv2.imread(projpath.as_posix(), cv2.IMREAD_ANYDEPTH)
                plotter.set_channel(c, proj)
            plotter.plot()
            plotter.plot_composite()
            if disp:
                plt.show()
            if cc.export_flags['figure']:
                plotter.save_fig()

            plt.close('all')
