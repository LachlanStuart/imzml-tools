import argparse
import xml.etree.ElementTree as etree
from shutil import copyfile

import numpy as np
from skimage import measure

# This version keeps all imzML metadata, but doesn't filter out redundant information from the
# .ibd file


def get_spectrum_coords(s):
    x = s.find(".//{http://psi.hupo.org/ms/mzml}cvParam[@name='position x']").get('value')
    y = s.find(".//{http://psi.hupo.org/ms/mzml}cvParam[@name='position y']").get('value')
    return int(x), int(y)


parser = argparse.ArgumentParser()
parser.add_argument('input', help='input imzML file')
args = parser.parse_args()
assert args.input.endswith('.imzML')

# Open original file
et = etree.parse(args.input)

# find all spectrum coordinates
spectrumList = et.find('.//{http://psi.hupo.org/ms/mzml}spectrumList')
spectra = et.findall('.//{http://psi.hupo.org/ms/mzml}spectrum')
all_coords = [get_spectrum_coords(s) for s in spectra]
spectrum_by_coord = dict(zip(all_coords, spectra))

# Plot pixel coordinates & split into regions
bitmap = np.zeros(np.max(all_coords, axis=0) + 1)
for c in all_coords:
    bitmap[c] = 1
region_labels, num_regions = measure.label(bitmap, return_num=True)

# Write each region to a separate imzML file
for i in range(num_regions):
    for c in spectrumList.getchildren():
        spectrumList.remove(c)

    region_coords = np.argwhere(region_labels == i + 1)
    for x,y in region_coords:
        spectrumList.append(spectrum_by_coord[(x,y)])

    mid_x, mid_y = map(int, np.mean(region_coords, axis=0))

    filename = args.input.replace('.imzML', f'_{mid_x}_{mid_y}.imzML')
    print(f'Writing {len(region_coords)} spectra to {filename}')
    et.write(filename)

    ibd_old = args.input.replace('.imzML', '.ibd')
    ibd_new = filename.replace('.imzML', '.ibd')
    print(f'Copying {ibd_old} to {ibd_new}')
    copyfile(ibd_old, ibd_new)
