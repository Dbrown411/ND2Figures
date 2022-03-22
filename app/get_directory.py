from tkinter import filedialog
from tkinter import *
from app.config import ConfigContainer as cc
from pathlib import Path


def create_dir_select(parent) -> filedialog:
    window_title = 'Select directory with .nd2 Files'
    return (filedialog.askdirectory(parent=parent,
                                    initialdir=cc.ihc_dir,
                                    title=window_title))


def get_user_path() -> Path:
    root = Tk()
    root.withdraw()
    if root._windowingsystem == 'win32':
        top = Toplevel(root)
        top.iconify()
        folder_selected = create_dir_select(top)
        top.destroy()
    else:
        folder_selected = create_dir_select(root)
    root.grab_release()
    root.destroy()
    if folder_selected != '':
        directory = folder_selected
    return Path(directory)