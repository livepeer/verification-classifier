#!/usr/bin/env python3
#
# Calculate SSIM/PSNR from video
#
# See also: https://github.com/stoyanovgeorge/ffmpeg/wiki/How-to-Compare-Video
#
# Author: Werner Robitza
# License: MIT

from functools import reduce
import argparse
import subprocess
import os
import json
import sys
import tempfile
from pprint import pprint
import pandas as pd
from platform import system as _current_os

# from .__init__ import __version__ as version

ALLOWED_SCALERS = ["fast_bilinear",
                   "bilinear",
                   "bicubic",
                   "experimental",
                   "neighbor",
                   "area",
                   "bicublin",
                   "gauss",
                   "sinc",
                   "lanczos",
                   "spline"]
CUR_OS = _current_os()
IS_WIN = CUR_OS in ['Windows', 'cli']
IS_NIX = (not IS_WIN) and any(
    CUR_OS.startswith(i) for i in
    ['CYGWIN', 'MSYS', 'Linux', 'Darwin', 'SunOS', 'FreeBSD', 'NetBSD'])
NUL = 'NUL' if IS_WIN else '/dev/null'

def get_brewed_model_path():
    """
    Hack to get path for VMAF model from Linuxbrew
    """
    stdout, _ = run_command(["brew", "--prefix", "libvmaf"])
    cellar_path = stdout.strip()

    model_path = os.path.join(cellar_path, "share", "model")

    return model_path


def print_stderr(msg):
    print(msg, file=sys.stderr)


def run_command(cmd, dry_run=False, verbose=False):
    """
    Run a command directly
    """
    if dry_run or verbose:
        print_stderr("[cmd] " + " ".join(cmd))
        if dry_run:
            return

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        return stdout.decode("utf-8"), stderr.decode("utf-8")
    else:
        print_stderr("[error] running command: {}".format(" ".join(cmd)))
        print_stderr(stderr.decode("utf-8"))
        sys.exit(1)

def calc_ssim_psnr(ref, dist, scaling_algorithm="bicubic", dry_run=False, verbose=False):
    psnr_data = []
    ssim_data = []

    if scaling_algorithm not in ALLOWED_SCALERS:
        print_stderr(f"Allowed scaling algorithms: {ALLOWED_SCALERS}")

    try:
        temp_dir = tempfile.gettempdir()

        temp_file_name_ssim = os.path.join(
            temp_dir, next(tempfile._get_candidate_names()) + "-ssim.txt"
        )
        temp_file_name_psnr = os.path.join(
            temp_dir, next(tempfile._get_candidate_names()) + "-psnr.txt"
        )

        if verbose:
            print_stderr(f"Writing temporary SSIM information to: {temp_file_name_ssim}")
            print_stderr(f"Writing temporary PSNR information to: {temp_file_name_psnr}")

        filter_chains = [
            f"[1][0]scale2ref=flags={scaling_algorithm}[dist][ref]",
            "[dist]split[dist1][dist2]",
            "[ref]split[ref1][ref2]",
            f"[dist1][ref1]psnr={temp_file_name_psnr}",
            f"[dist2][ref2]ssim={temp_file_name_ssim}"
        ]

        cmd = [
            "ffmpeg", "-nostdin", "-y",
            "-threads", "1",
            "-i", ref,
            "-i", dist,
            "-filter_complex",
            ";".join(filter_chains),
            "-an",
            "-f", "null", NUL
        ]

        run_command(cmd, dry_run, verbose)

        if not dry_run:
            with open(temp_file_name_psnr, "r") as in_psnr:
                # n:1 mse_avg:529.52 mse_y:887.00 mse_u:233.33 mse_v:468.25 psnr_avg:20.89 psnr_y:18.65 psnr_u:24.45 psnr_v:21.43
                lines = in_psnr.readlines()
                for line in lines:
                    line = line.strip()
                    fields = line.split(" ")
                    frame_data = {}
                    for field in fields:
                        k, v = field.split(":")
                        frame_data[k] = round(float(v), 3) if k != "n" else int(v)
                    psnr_data.append(frame_data)

            with open(temp_file_name_ssim, "r") as in_ssim:
                # n:1 Y:0.937213 U:0.961733 V:0.945788 All:0.948245 (12.860441)\n
                lines = in_ssim.readlines()
                for line in lines:
                    line = line.strip().split(" (")[0]  # remove excess
                    fields = line.split(" ")
                    frame_data = {}
                    for field in fields:
                        k, v = field.split(":")
                        if k != "n":
                            # make psnr and ssim keys the same
                            k = "ssim_" + k.lower()
                            k = k.replace("all", "avg")
                        frame_data[k] = round(float(v), 3) if k != "n" else int(v)
                    ssim_data.append(frame_data)

    except Exception as e:
        raise e
    finally:
        if os.path.isfile(temp_file_name_psnr):
            os.remove(temp_file_name_psnr)
        if os.path.isfile(temp_file_name_ssim):
            os.remove(temp_file_name_ssim)

    return {
        'ssim': ssim_data,
        'psnr': psnr_data
    }
