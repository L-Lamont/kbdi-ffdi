#!/usr/bin/env python

import argparse
import time
import os
import glob

from kbdiffdi import *

def __parse_args():
    parser = argparse.ArgumentParser(description="""Script for computing kbdi and ffdi
                                                    """)

    parser.add_argument('-i',
                        '--input',
                        dest='input_filename',
                        required=True,
                        type=str,
                        help="input CSV file, OR a directory of CSV files to "
                             "process in batch")
    parser.add_argument('-o',
                        '--output',
                        dest='output_filename',
                        required=True,
                        type=str,
                        help="output CSV file (per-file mode), OR an output "
                             "directory when the input is a directory")
    parser.add_argument('-s',
                        '--spinup-days',
                        dest='spinup_days',
                        type=int,
                        default=0,
                        help="number of leading days to drop from the output CSV as "
                             "KBDI spin-up. These days are still computed (the cumulative "
                             "index needs them) but not written, since the index starts "
                             "from an arbitrary KBDI=0 and the early values are unreliable. "
                             "Default 0 (write everything).")
    parser.add_argument('-v',
                        '--verbose',
                        dest='verbose',
                        action='store_true',
                        default=True)
    args = parser.parse_args()

    if args.verbose:
        print("------------ User Input ----------------")
        print('input file:\t' + args.input_filename)
        print('output file:\t' + args.output_filename)
        print('spin-up days:\t' + str(args.spinup_days))
        print()

    return args

def __build_jobs(input_path, output_path):
    """
    Work out the list of (input_file, output_file) pairs to process.

    If input_path is a directory, every *.csv in it is processed in batch and
    output_path is treated as a directory: each result keeps its input's
    basename. Otherwise this is per-file mode and the single pair
    (input_path, output_path) is returned.

    Returns a list of (input_file, output_file) tuples (empty if nothing to do).
    """
    if os.path.isdir(input_path):
        csvs = sorted(glob.glob(os.path.join(input_path, "*.csv")))
        return [(f, os.path.join(output_path, os.path.basename(f))) for f in csvs]
    return [(input_path, output_path)]

def __check_args(args):
    args_are_good = True

    if not os.path.exists(args.input_filename):
        print("[ERROR] - input file: " + args.input_filename + " doesn't exist")
        print(" ... check to make sure the filename and filepath are correct")
        print()
        args_are_good = False
        return args_are_good

    if os.path.isdir(args.input_filename):
        # --- directory / batch mode ---
        if not glob.glob(os.path.join(args.input_filename, "*.csv")):
            print("[ERROR] - input directory: " + args.input_filename + " contains no .csv files")
            print()
            args_are_good = False
        # the output is a directory here; create it if it doesn't exist
        if not os.path.exists(args.output_filename):
            os.makedirs(args.output_filename, exist_ok=True)
            print("[INFO] created output directory: " + args.output_filename)
        elif not os.path.isdir(args.output_filename):
            print("[ERROR] - input is a directory, so output: " + args.output_filename + " must be a directory too")
            print()
            args_are_good = False
    else:
        # --- per-file mode ---
        # dirname is "" for a bare filename / relative path with no directory
        # part, which means "current directory" - treat that as valid.
        output_dir = os.path.dirname(args.output_filename) or "."
        if not os.path.exists(output_dir):
            print("[ERROR] - output filepath: " + output_dir + " doesn't exist")
            print(" ... you tried to save the output file to a directory that doesn't exist. \n check to make sure the filepath is set correctly")
            print()
            args_are_good = False

    if not args_are_good:
        print(" ------- PROGRAM FAILED :( -------")

    return args_are_good

def run_kbdi_ffdi(input_filename, output_filename, spinup_days=0):
    print('[INFO] reading input')
    rain, temp, relhum, wind = input_output.load_csv(input_filename)

    print("[INFO] computing KBDI")
    kbdi = indices.KBDI()
    out_kbdi = kbdi.fit(temp, rain)

    print("[INFO] computing FFDI")
    ffdi = indices.FFDI()
    out_ffdi, out_df = ffdi.fit(out_kbdi, rain, temp, wind, relhum)

    print("[INFO] writing output to .csv")
    if spinup_days:
        print("       (dropping first %d day(s) as spin-up)" % spinup_days)
    input_output.write_csv(input_filename, output_filename, out_kbdi, out_ffdi, out_df, spinup_days=spinup_days)


def main():
    print("\n\
 _   ______________ _____     __________________ _____ \n\
| | / /| ___ \  _  \_   _|    |  ___|  ___|  _  \_   _|\n\
| |/ / | |_/ / | | | | |______| |_  | |_  | | | | | |  \n\
|    \ | ___ \ | | | | |______|  _| |  _| | | | | | |  \n\
| |\  \| |_/ / |/ / _| |_     | |   | |   | |/ / _| |_ \n\
\_| \_/\____/|___/  \___/     \_|   \_|   |___/  \___/ \n\
                                                       ")
    start_time = time.time()
    print('\nStart date & time --- (%s)\n' % time.asctime(time.localtime(time.time())))

    args = __parse_args()

    if __check_args(args):
        jobs = __build_jobs(args.input_filename, args.output_filename)
        batch = len(jobs) > 1 or os.path.isdir(args.input_filename)
        if batch:
            print("[INFO] batch mode: %d file(s) to process\n" % len(jobs))
        failures = 0
        for n, (in_file, out_file) in enumerate(jobs, 1):
            if batch:
                print("===== [%d/%d] %s -> %s =====" % (n, len(jobs), in_file, out_file))
            try:
                run_kbdi_ffdi(in_file, out_file, args.spinup_days)
            except Exception as exc:
                # in batch mode, one bad file shouldn't kill the whole run
                failures += 1
                print("[ERROR] failed on %s: %s" % (in_file, exc))
                if not batch:
                    raise
            print()
        if batch:
            print("[INFO] finished: %d succeeded, %d failed" % (len(jobs) - failures, failures))

    tot_sec = time.time() - start_time
    minutes = int(tot_sec // 60)
    sec = tot_sec % 60
    print('\nEnd data & time -- (%s)\nTotal run-time -- (%d min %f sec)\n' %
        (time.asctime(time.localtime(time.time())), minutes, sec))

if __name__ == "__main__":
    main()
