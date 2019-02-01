#!/usr/bin/env python3

import argparse
import os
import re
from glob import glob

from tqdm import tqdm

from ucca.convert import to_text
from ucca.ioutil import file2passage, get_passages_with_progress_bar

desc = """Parses files in UCCA standard format, and writes as text files or a text file with a line per passage."""


def numeric(x):
    try:
        return tuple(map(int, re.findall("\d+", x)))
    except ValueError:
        return x


def write_text(passage, f, lang):
    for line in to_text(passage, lang=lang):
        print(line, file=f)


def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    if args.join:
        out_file = os.path.join(args.outdir, args.join)
        with open(out_file, "w", encoding="utf-8") as f:
            for passage in get_passages_with_progress_bar(sorted(args.filenames, key=numeric), desc="Converting"):
                write_text(passage, f, lang=args.lang)
        print("Wrote '%s'." % out_file)
    else:  # one file per passage
        for pattern in args.filenames:
            for filename in tqdm(glob(pattern) or [pattern], desc="Converting", unit=" passages"):
                passage = file2passage(filename)
                basename = os.path.splitext(os.path.basename(filename))[0]
                with open(os.path.join(args.outdir, basename + ".txt"), "w", encoding="utf-8") as f:
                    write_text(passage, f, lang=args.lang)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("filenames", nargs="+", help="passage file names to convert")
    argparser.add_argument("-o", "--outdir", default=".", help="output directory")
    argparser.add_argument("-l", "--lang", default="en", help="language two-letter code for sentence model")
    argparser.add_argument("-j", "--join", help="write just one text file with this name, with one line per passage")
    main(argparser.parse_args())
