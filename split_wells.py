import argparse
import numpy as np
from skimage import measure
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter



def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('input', help='input imzML file')
    arg_parser.add_argument(
        '--keep-coords',
        action='store_true',
        help="Pass this to keep the original coordinates. If not passed, each well will be "
             "adjusted so that the top-left is at coordinate 1,1,1. Useful if the files will later "
             "be merged back into one"
    )

    args = arg_parser.parse_args()
    assert args.input.endswith('.imzML')

    return args


def split_imzml_file(in_file, keep_coords):
    # Open original file
    imzml_parser = ImzMLParser(in_file)
    coords = np.array(imzml_parser.coordinates)
    coord_to_idx = {tuple(c): i for i, c in enumerate(coords)}

    # Plot pixel coordinates & split into regions
    bitmap = np.zeros(np.max(coords, axis=0) + 1)
    for c in coords:
        bitmap[tuple(c)] = 1
    region_labels, num_regions = measure.label(bitmap, return_num=True)

    # Write each region to a separate imzML file
    for i in range(num_regions):
        region_coords = np.argwhere(region_labels == i + 1)
        mid_x, mid_y = np.mean(region_coords, axis=0)[:2].astype('i')

        out_file = in_file.replace('.imzML', f'_{mid_x}_{mid_y}.imzML')
        print(f'Writing {len(region_coords)} spectra to {out_file}')

        with ImzMLWriter(out_file) as writer:
            offset = (1 - np.min(region_coords, axis=0))
            if keep_coords:
                offset = np.zeros_like(offset)

            for coord in region_coords:
                mzs, ints = imzml_parser.getspectrum(coord_to_idx[tuple(coord)])
                writer.addSpectrum(mzs, ints, coord + offset)


if __name__ == '__main__':
    args = parse_args()
    split_imzml_file(args.input, args.keep_coords)