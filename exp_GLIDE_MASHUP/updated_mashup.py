#!/cluster/tufts/cowenlab/.envs/denoise/bin/python
import os
import sys
sys.path.append("./")
import numpy as np
import json as js
from utils import compute_mashup_updated
import argparse

def getparams():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder")
    parser.add_argument("--dims", default = 1000, type = int)
    parser.add_argument("--verbose", default = False, action = "store_true")
    parser.add_argument("--const", default = 2, type = float)
    return parser.parse_args()


def main(args):
    def log(strng):
        if args.verbose:
            print(strng)

    network_files = [f"{args.input_folder}/{f}" for f in os.listdir(args.input_folder) if (f.endswith("npy") and not f.startswith("OUT"))]
    As            = [np.load(f) for f in network_files]
    E             = compute_mashup_updated(As, reduced_dim = args.dims, const_param = args.const)
    
    out_file      = f"{args.input_folder}/OUT-dim_{args.dims}_const_{args.const}.npy"

    np.save(out_file, E)

if __name__ == "__main__":
    main(getparams())
