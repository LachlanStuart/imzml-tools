import argparse
import numpy as np
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter


def fix_imzml(input_file, output_file, polarity):
    parser = ImzMLParser(input_file)
    assert hasattr(parser, 'polarity'), 'Old version of pyimzml - please run "pip install -U pyimzml"'
    if polarity is None:
        if parser.polarity in ('positive', 'negative'):
            polarity = parser.polarity
        else:
            print('No polarity found/specified. Assuming positive mode')
            polarity = 'positive' # Just guess

    coordinates = np.array(parser.coordinates)[:, :2]
    if (coordinates < 1).any():
        # MSiReader doesn't support zero/negative coordinates. Move the top-most/left-most coordinate to 1,1
        coordinates -= coordinates.min(axis=0) - 1

    if output_file is None:
        output_file = input_file[:-len('.imzML')] + '_fixed.imzML'

    with ImzMLWriter(output_file, polarity=polarity) as writer:
        for idx, (x, y) in enumerate(coordinates):
            mzs, ints = parser.getspectrum(idx)
            writer.addSpectrum(mzs, ints, (x, y, 1))

def parse_args():
    arg_parser = argparse.ArgumentParser(description='Fixes ImzML file metadata (polarity, coordinates) so that MSiReader can open it')
    arg_parser.add_argument('input', help='input imzML file')
    arg_parser.add_argument('-o', '--output', help='output imzML file (default: same as input with _fixed suffix)')
    arg_parser.add_argument(
        '--polarity',
        choices=['positive', 'negative'],
        help="Optional. Specify this to override the polarity"
    )

    args = arg_parser.parse_args()
    assert args.input.endswith('.imzML')

    return args

if __name__ == '__main__':
    args = parse_args()
    fix_imzml(args.input, args.output, args.polarity)