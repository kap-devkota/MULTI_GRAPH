import os
import sys
sys.path.append("./")
from phate_utils import compute_potential_dist
import argparse
import pandas as pd
import seaborn as sns
import json
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_file", help = "The input network file", required = True)
    parser.add_argument("--output_prefix", help = "Output prefix", default = "")
    parser.add_argument("--n_dims", help = "PHATE dimension", default = 100, type = float)
    parser.add_argument("--timesteps", help = "Timesteps", default = 5, type = int)
    parser.add_argument("--verbose", default = False, action = "store_true")
    return parser.parse_args()


def get_adjacency(filename):
    df    = pd.read_csv(filename, delim_whitespace = True, header = False)
    nodes = set(df[0]).union(set(df[1]))
    nodemap = {k: i for i, k in enumerate(nodes)}

    df_annotate = df.replace({0:nodemap, 1:nodemap})

    A     = np.zeros((len(nodes), len(nodes)))
    for p,q, w in df_annotate.to_records(index = False):
        A[p, q] = w
        A[q, p] = w
    return A, nodemap
    
def main(args):
    def log(strng):
        if args.verbose:
            print(strng)
    A, nmap = get_adjacency(args.network_file)
    X       = compute_potential_dist(A, args.timesteps, args.n_dims)

    output_emb, output_json = [f"{output_prefix}_dim_{args.n_dims}_t_{args.timesteps}.{k}"
                               for k in ["npy", "json"]]
    np.save(output_emb, X)
    with open(output_json, "w") as oj:
        json.dump(nmap, oj)


    if args.n_dims == 2:
        """
        Perform SNS visualization
        """
        dframe = pd.DataFrame(columns = ["p0", "p1"])
        dframe["p0"] = X[0]
        dframe["p1"] = X[1]
        sns.set_theme("blackgrid")
        sns.relplot(data = dframe, x = "p0", y = "p1")
        plt.savefig(f"{output_prefix}_dim_{args.n_dims}_t_{args.timesteps}.png")


if __name__ == "__main__":
    main(get_args())
