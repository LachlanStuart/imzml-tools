from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
files_and_offsets = [
('top_left.imzML', (0, 0)),
('top_right.imzML', (50, 0)),
# etc.
]

with ImzMLWriter('output.imzML') as writer:
    for file, (offset_x, offset_y) in files_and_offsets:
        parser = ImzMLParser(file)
        for idx, (x, y, z) in enumerate(parser.coordinates):
            mzs, ints = parser.getspectrum(idx)
            writer.addSpectrum(mzs, ints, (x+offset_x, y+offset_y, z))