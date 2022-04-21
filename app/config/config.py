from pathlib import Path


class ConfigContainer:
    config_path = Path(__file__).parent
    root_path = config_path.parent.parent
    cache_path = root_path / "cache"
    LASTDIR = cache_path / "LASTDIRECTORY.txt"
    laptop = Path(rf"C:\Users\dillo\OneDrive\Desktop\OneDrive Backup")
    ihc_dir = laptop / "Thesis/Data/IHC"

    groupby_slice = (0, -1)
    identifier_slice = (0, -1)

    calc_proj = True
    create_fig = True

    export_flags = {
        'raw': True,
        'proj': calc_proj,
        'figure': create_fig,
        'offset': True
    }

    channel_to_proj_map = {
        '405': 'max',
        '488': 'mean',
        '561': 'mean',
        '640': 'max'
    }
    show_progress = True