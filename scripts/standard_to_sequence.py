#!/usr/bin/python3


desc = """Parses an XML in UCCA standard format, and writes it as a linearized sequence.
"""

import argparse
import sys

import ucca.convert
from util import file2passage


def main():
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('filenames', nargs='+', help="XML file names to convert")
    parser.add_argument('-o', '--outfile', help="output file for sequence")
    args = parser.parse_args()

    handle = open(args.outfile, 'w') if args.outfile else sys.stdout
    for filename in args.filenames:
        sys.stderr.write("Reading passage '%s'...\n" % filename)
        passage = file2passage(filename)
        output = ucca.convert.to_sequence(passage)
        handle.write(output)
        handle.write("\n")
    if handle is not sys.stdout:
        handle.close()

    sys.exit(0)


if __name__ == '__main__':
    main()