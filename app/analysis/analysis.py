from skimage import restoration
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from app.utilities import *
import warnings


def thresh_plot(im, label: str = ''):
    fig, ax = plt.subplots()
    ax.imshow(im)
    if label != '':
        plt.suptitle(label)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.show()
        plt.close()
    return fig, ax


def framestack_to_vol(framestack: "list[NDArray]") -> "NDArray":
    vol = np.dstack(framestack)
    vol = np.rollaxis(vol, -1)
    return vol


def get_projection(vol, method='mean'):
    meth_map = {
        'mean':
        lambda x: np.nanmean(np.int32(x)[0:, :, :], axis=0, dtype=np.int32),
        'med': lambda x: np.nanmedian(x[0:, :, :], axis=0),
        'max': lambda x: np.nanmax(x[0:, :, :], axis=0),
        'sum':
        lambda x: np.nansum(np.int32(x)[0:, :, :], axis=0, dtype=np.int32),
    }
    proj = meth_map[method](vol)
    # fig, ax = plt.subplots()
    # ax.hist(proj.ravel(), bins=256, range=(0, 256))
    proj = np.clip(proj, 0, 65535)
    # ax.hist(proj.ravel(), bins=256, range=(0, 256))
    # ax.set_yscale('log')
    # plt.show()
    return np.uint16(proj)


def find_frame_maxes(frames=[]):
    norm_against = [np.max(x) for x in frames]
    overall_max = np.max(norm_against)
    norm_against = [np.divide(x, overall_max) for x in norm_against]
    norm_against = [int(np.round(x * 255)) for x in norm_against]
    return norm_against


def plot_result(image, fg, bg):
    fg = np.nan_to_num(fg)
    bg = np.nan_to_num(bg)
    immax, bgmax = find_frame_maxes([image, bg])
    fig, ax = plt.subplots(nrows=1, ncols=3)
    bg_frame = normalize_frame(bg, bgmax)
    fg_frame = normalize_frame(fg, immax)
    image_frame = normalize_frame(image, immax)

    ax[0].imshow(bg_frame, cmap='Greens', vmin=0, vmax=immax)
    ax[0].set_title('Background')
    ax[0].axis('off')

    ax[1].imshow(fg_frame, cmap='Greens', vmin=0, vmax=immax)
    ax[1].set_title('Foreground')
    ax[1].axis('off')

    ax[2].imshow(image_frame, cmap='Greens', vmin=0, vmax=immax)
    ax[2].set_title('Original')
    ax[2].axis('off')

    fig.tight_layout()
    return fig


def filter_vol(vol):
    radz = 5
    radxy = 5
    background = restoration.rolling_ball(vol,
                                          kernel=restoration.ellipsoid_kernel(
                                              (1, radxy * 2, radxy * 2),
                                              radxy * 2),
                                          nansafe=True)
    for i in range(vol.shape[0]):
        plot_result(vol[i, :, :], background[i, :, :])
        plt.show()
    return vol - background


def offset_projection(proj, new_0):
    original = np.array(proj, dtype=np.int32)
    max16 = np.max(original)
    newmax16 = max16 - new_0
    newImage = np.array(original - new_0)
    newImage = newImage.clip(min=0)
    offset_proj = cv2.normalize(newImage,
                                None,
                                0,
                                newmax16,
                                cv2.NORM_MINMAX,
                                dtype=cv2.CV_16U)
    return offset_proj


def get_max(data, saturated: int = 0.2):
    perc = 100 - saturated
    result = np.percentile(np.ravel(data), perc, interpolation='nearest')
    return result


def identify_20X_foreground(img, **kwargs):
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    max_val = get_max(blur, saturated=70)
    blur = np.clip(blur, 0, max_val)
    thresh = cv2.normalize(blur,
                           None,
                           0,
                           255,
                           cv2.NORM_MINMAX,
                           dtype=cv2.CV_8U)
    thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 7, 1)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    kernel = np.ones((10, 10), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh,
                        kernel,
                        iterations=5 * int(max(img.shape) / 512))
    fig, ax = thresh_plot(thresh, kwargs['label'])
    return thresh


def identify_4X_foreground(img, **kwargs):
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    max_val = get_max(blur, saturated=20)
    blur = np.clip(blur, 0, max_val)
    thresh = cv2.normalize(blur,
                           None,
                           0,
                           255,
                           cv2.NORM_MINMAX,
                           dtype=cv2.CV_8U)
    thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 7, 1)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    kernel = np.ones((10, 10), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh,
                        kernel,
                        iterations=5 * int(max(img.shape) / 512))
    # fig, ax = thresh_plot(thresh, kwargs['label'])

    return thresh


##Main data analysis pipeline
def analyze_fstack(vol, proj_type, offset=True, **kwargs):
    try:
        label = kwargs['label']
    except KeyError:
        label = ''
    proj = get_projection(vol, proj_type)
    if not offset:
        return proj, ()
    # frame = normalize_frame(proj)

    mask_fg = identify_20X_foreground(proj, **{'label': label})
    # mask_fg = identify_4X_foreground(proj, **{'label': label})

    mask_bg = cv2.bitwise_not(mask_fg)
    mask16_fg = cv2.normalize(mask_fg,
                              None,
                              0,
                              65535,
                              cv2.NORM_MINMAX,
                              dtype=cv2.CV_16U)
    mask16_bg = cv2.normalize(mask_bg,
                              None,
                              0,
                              65535,
                              cv2.NORM_MINMAX,
                              dtype=cv2.CV_16U)
    foreground = cv2.bitwise_and(proj, mask16_fg)
    foreground = np.array(foreground, dtype=float)
    foreground[foreground == 0] = np.nan
    background = cv2.bitwise_and(proj, mask16_bg)
    background = np.array(background, dtype=float)
    background[background == 0] = np.nan

    mean_background = np.nanmean(background)

    offset = int(round(mean_background * 1.5))
    if proj_type == 'max':
        offset *= 2
    offset_proj = offset_projection(proj, offset)

    return offset_proj, (foreground, background), (mask_fg, mask_bg)
