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


def initialize_folders(directory: Path, clear: bool,
                       folders_with_data: list) -> ExportLocation:

    ##Clear previous results
    if cc.compile_all:
        export_paths = create_folders(directory, cc.export_flags, clear=clear)
    elif clear:
        export_paths = [
            create_folders(folder, cc.export_flags, clear=True)
            for folder in folders_with_data
        ][0]
    return export_paths


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

    export_locations = initialize_folders(directory, clear, folders_with_data)
    plotter = SamplePlotter(norm_across_samples=True,
                            norm_across_wavelengths=False)
    folder_count = 0
    samples = {}
    channel_maxes = {}
    image_outcomes_comp = {}
    unique_samples = []
    for folder in folders_with_data:
        folder_count += 1

        if not cc.compile_all:
            export_locations = create_folders(folder, cc.export_flags)

        unique_samples.extend(match_scans(folder, groupby=groupby))
    if cc.show_progress:
        pbar = tqdm(unique_samples[:2])
    else:
        pbar = unique_samples[:2]
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
        for i, (c, fstack) in enumerate(named_channels):
            projection_type = cc.channel_to_proj_map[c]
            channel_outname = f'{sample.name}-{c}nm'
            projection_outname = f"{channel_outname}-{projection_type.upper()}"
            if cc.export_flags['raw']:
                imageio.mimwrite(
                    export_locations.raw / f'{channel_outname}.tiff', fstack)
            vol = framestack_to_vol(fstack)
            ##Point at which custom analysis can be performed
            ##Function should take a 3d np.vol of uint16 and return a 2d array of uint16 for display
            ##
            proj, (fg, bg), (mask_fg,
                             mask_bg) = analyze_fstack(vol,
                                                       projection_type,
                                                       offset=offset)
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
                # plt.show()
                plt.close(fig_fgbg)
            write_proj(proj, export_locations.proj, projection_outname)

            chans.append(
                (c, export_locations.proj / f"{projection_outname}.tif"))
        image_outcomes_comp[sample.name] = image_outcomes
        samples[sample] = (export_locations.fig, chans)
    with open('analyzed.json', 'w') as fp:
        json.dump(image_outcomes_comp, fp)

    if cc.create_fig:
        plotter.channel_maxes = channel_maxes
        for sample, (export_locations.fig, channels) in samples.items():
            plotter.fig_folder = export_locations.fig
            plotter.sample = sample
            plotter.init_plot()
            for c, projpath in channels:
                proj = cv2.imread(projpath, cv2.IMREAD_ANYDEPTH)
                plotter.set_channel(c, proj)
            plotter.plot()
            plotter.plot_composite()
            if disp:
                plt.show()
            if cc.export_flags['figure']:
                plotter.save_fig()

            plt.close('all')
