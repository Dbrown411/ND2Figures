Functions/Use

1. Recursively crawl parent directory,
   find all .nd2 file types,
   optional compilation of different files using groupby
   --groupby tup(i,j) of filename split by (-|\_|.)

2. (Optional) export a 16bit .tiff stack of experiment

3. (Optional) Identify metadata and create/export 16bit projections
   with optional filtering/thresholding
   -By default assumes DAPI as 405nm
   and creates a maximum intensity projection for more visible
   counterstain
   -other hardcoded channels (480nm,561nm,640nm) will be
   mean intensity projections more suitable for quantification

4. (Optional) Convert projections to normalized 8bit RGB images,
   map wavelengths to colors and labeled proteins,
   Plot in order of descending wavelength, and add
   a composite image at the end.
   Add scalebar on first image based on metadata
   Add labels for each channel on each image
   Size figure correctly in order to export figure at original resolution for each channel
   if aspect ratio is >1:
   Arranges panels vertically
   if aspect ratio is <=1:
   Arranges horizontally
