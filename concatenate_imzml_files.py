import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import re
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description='Merges spectra from multiple overlapping imzML files that cover different m/z ranges'
    )
    arg_parser.add_argument(
        'inputs', nargs='+',
        help='''input imzML file paths, with optional offset and m/z range
        e.g.
        my_file.imzML - Use my_file.imzML, autodetecting the m/z splitting point
        my_file.imzML@3,2 - shift my_file.imzML by 3 pixels horizontally and 2 pixels vertically
        my_file.imzML:100-300 - take only the peaks between 100-300 m/z in my_file.imzML 
        my_file.imzML@3,2:100-300 - combine the above two transformations''')

    arg_parser.add_argument('--output', help='Output file path')

    args = arg_parser.parse_args()

    # Parse the input paths/offsets/mz-ranges
    input_paths = []
    offsets = []
    mz_ranges = []
    for input_spec in args.inputs:
        match = re.match(r'^(.*\.imzML)(?:@(\d+),(\d+))?(?::([\d\.]+)-([\d\.]+))?$', input_spec, flags=re.IGNORECASE)
        assert match, f'Could not understand input {input_spec}'

        path = match[1]
        x_offset = match[2]
        y_offset = match[3]
        mz_lo = match[4]
        mz_hi = match[5]
        input_paths.append(Path(path))
        offsets.append((int(x_offset or 0), int(y_offset or 0)))
        mz_ranges.append((
            float(mz_lo) if mz_lo else 'auto',
            float(mz_hi) if mz_hi else 'auto',
        ))
    args.input_paths = input_paths
    args.offsets = offsets
    args.mz_ranges = mz_ranges

    if not args.output:
        args.output = input_paths[0].with_name(f'{input_paths[0].stem}_concatenated.imzML')

    return args


def get_mz_range(parser):
    mzs = np.concatenate([parser.getspectrum(i)[0] for i in range(min(100, len(parser.coordinates)))])
    # Round outwards to nearest multiple of 10
    return np.round(np.min(mzs), -1), np.round(np.max(mzs), -1)


def concat_imzml_files(input_paths, offsets, input_mz_ranges, output_path):
    parsers = []
    for input_path in input_paths:
        print(f'Parsing imzML file for {input_path}')
        parsers.append(ImzMLParser(str(input_path)))
    full_mz_ranges = [get_mz_range(p) for p in parsers]
    order = np.argsort(np.mean(full_mz_ranges, axis=1))

    # Re-order all inputs
    input_paths = np.array(input_paths)[order]
    parsers = np.array(parsers)[order]
    offsets = np.array(offsets)[order]
    input_mz_ranges = np.array(input_mz_ranges, dtype='O')[order]
    full_mz_ranges = np.array(full_mz_ranges)[order]

    # Fill in "auto" values in input_mz_ranges
    mz_ranges = []
    for i, ((mz_lo, mz_hi), (full_lo, full_hi)) in enumerate(zip(input_mz_ranges, full_mz_ranges)):
        if mz_lo == 'auto':
            if i == 0:
                mz_lo = None  # No lower dataset - use full lower bound
            elif input_mz_ranges[i - 1][1] != 'auto':
                mz_hi = input_mz_ranges[i - 1][1] # Use below ds's specified bound
            else:
                # There's another DS below, use the midpoint of the overlapping area
                # Add 0.5 to get into the least crowded part of the spectrum
                mz_lo = np.round((full_mz_ranges[i - 1][1] + full_lo) / 2) + 0.5

        if mz_hi == 'auto':
            if i == len(input_mz_ranges) - 1:
                mz_hi = None  # No higher dataset - use full higher bound
            elif input_mz_ranges[i + 1][0] != 'auto':
                mz_hi = input_mz_ranges[i + 1][0]  # Use above ds's specified bound
            else:
                # There's another DS above, use the midpoint of the overlapping area
                # Add 0.5 to get into the least crowded part of the spectrum
                mz_hi = np.round((full_mz_ranges[i + 1][0] + full_hi) / 2) + 0.5

        mz_ranges.append((mz_lo, mz_hi))

        print(f'{input_paths[i].name} detected m/z range: {full_lo}-{full_hi}. Taking range {mz_lo or ""}-{mz_hi or ""}')

    # Read all imzML files and extract the m/z ranges of interest
    # Store in dicts indexed by the output coordinate
    output_mzs = defaultdict(list)
    output_ints = defaultdict(list)
    for input_path, parser, offset, (mz_lo, mz_hi) in zip(input_paths, parsers, offsets, mz_ranges):
        print(f'Reading spectra for {input_path}')
        coords = np.array(parser.coordinates)[:, :2]
        coords -= np.min(coords, axis=0)
        coords += [offset]
        for i, (x, y) in enumerate(coords):
            mzs, ints = parser.getspectrum(i)
            mask = ints != 0
            if mz_lo is not None:
                mask &= mzs >= mz_lo
            if mz_hi is not None:
                mask &= mzs < mz_hi
            output_mzs[(x,y)].append(mzs[mask])
            output_ints[(x,y)].append(ints[mask])


    with ImzMLWriter(output_path) as writer:
        print(f'Writing to {output_path}')
        for x, y in sorted(output_mzs.keys()):
            mzs = np.concatenate(output_mzs[(x,y)])
            ints = np.concatenate(output_ints[(x,y)])
            writer.addSpectrum(mzs, ints, (x+1, y+1, 1))


if __name__ == '__main__':
    args = parse_args()
    concat_imzml_files(args.input_paths, args.offsets, args.mz_ranges, args.output)