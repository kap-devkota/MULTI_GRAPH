import argparse
import random
import numpy as np
import json
import os
import sys
sys.path.append("../scoring/")
import scoring
import predict
import ctypes
import graph_io
import pandas as pd
import re
import scipy.spatial.distance as spatial


def log_write(msg):
    with open(LOG_FILE, "w+") as logf:
        logf.write(msg)
    return

def log(msg):
    print(msg)

def create_predictor(E, params = {}, confidence = True):
    def predictor(training_labels):
        tlabels_f = lambda i: (training_labels[i] if i in training_labels else [])
        labels_dict = {}
        for i in range(E.shape[0]):
            l = tlabels_f(i)
            if len(l) != 0:
                labels_dict[i] = l
        return predict.perform_binary_OVA(E, labels_dict, params = params, clf_type = "LR", confidence = confidence)
    return predictor

def entrez_dict(IS_SYMBOL = True):
    if not IS_SYMBOL:
        ensp_dict = {}
        header    = True
        with open("../../datasets/STRING/9606.protein.info.v11.5.txt", "r") as of:
            for line in of:
                if header:
                    header = False
                    continue
                words = re.split("\t", line.strip())
                if len(words) >= 2 and not words[1].startswith("ENSG"):
                    ensp_dict[words[1]] = words[0]
            
    s_e_dict = {}
    with open("../../datasets/MISC/idmap.csv", "r") as of:
        header = True
        for line in of:
            if header:
                header = False
                continue
            words = re.split("\t", line.strip())
            if len(words) >= 2 and words[1] != "":
                if not IS_SYMBOL and words[0] in ensp_dict:
                    s_e_dict[ensp_dict[words[0]]] = int(words[1])
                else:
                    s_e_dict[words[0]] = int(words[1])
    rev_dict = {s_e_dict[k]: k for k in s_e_dict}
    return s_e_dict, rev_dict

"""
run using the command:
python classify.py --network=../output/DREAM/emb_1000.npy --json=../../dataset/human/DREAM/nodemap.json --dim=1000
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network")
    parser.add_argument("--json")
    parser.add_argument("--output")
    parser.add_argument("-v", action="store_true", help="Verbose mode")
    parser.add_argument("--minN", type = int, default = 50, help = "GO Label file minimum")
    parser.add_argument("--level", type = int, default = 5, help = "GO Label file maximum")
    parser.add_argument("--go_type", choices=["P", "F", "C"], default="F", help = "GO Type")
    parser.add_argument("--pr", default = False, action = "store_true")
    parser.add_argument("--is_ensp", action = "store_true", default = False)

    args     = parser.parse_args()
    verbose  = args.v
    network  = args.network
    njson    = args.json
    
    random.seed(42)

    log("Parsing Embedding")
    E           = np.load(network, allow_pickle = True)
    E           = np.absolute(E)
    
    print(E.shape)
    with open(njson, "r") as nj:
        node_map = json.load(nj)
        r_node_map    = {k: int(i) for i, k in node_map.items()}
    node_list    = list(range(len(node_map)))
    for k in node_map:
        node_list[node_map[k]] = k

    log("Parsed Embedding")

    print(len(node_list))
    n, _ = E.shape

    log("Building GO Annotations")

    GO_TYPE="molecular_function"
    if args.go_type == "P":
        GO_TYPE="biological_process"
    elif args.go_type == "C":
        GO_TYPE="cellular_component"
    filter_protein = {"namespace": GO_TYPE, "lower_bound": args.minN}
    filter_labels  = {"namespace": GO_TYPE, "min_level": args.level, "max_level": args.level}
    filter_parents = {"namespace": GO_TYPE}                
    s_entrez, entrez_s = entrez_dict(not args.is_ensp)
    e_symbols          = [s_entrez[k] for k in node_list if k in s_entrez]
    f_labels, labels_dict, parent_dict = graph_io.get_go_labels_and_parents("../../datasets/GO/go-basic.obo",
                                                                            "../../datasets/GO/gene2go",
                                                                            filter_protein, 
                                                                            filter_labels,
                                                                            filter_parents,
                                                                            e_symbols,
                                                                            anno_map = lambda x: entrez_s[x])
    proteins_to_go     = {}
    for l in labels_dict:
        prots = labels_dict[l]
        for p in prots:
            if p not in proteins_to_go:
                proteins_to_go[p] = []
            proteins_to_go[p] += [l]    
    labels = {i: proteins_to_go[node_list[i]] 
              for i in range(n) if node_list[i] in proteins_to_go}
    log("Completed Building GO Annotations")

    # Results
    results = {}
    
    kfold = 5
    namespace = "MF"
    if GO_TYPE == "biological_process":
        namespace = "BP"
    if GO_TYPE == "cellular_component":
        namespace = "CC"
    metric = scoring.kfoldcv_sim(kfold,
                                 labels,
                                 create_predictor(E),
                                 namespace = namespace,
                                 ci = 20)
    meth   = "RESNIK_SIM"
    for i in range(len(metric)):
        print(f"Fold {i + 1} {meth}: {metric[i]}")
    results["RESNICK"] = metric
    print(f"Mean {meth}: {np.mean(metric)}: Std: {np.std(metric)}")
    

    metric = scoring.kfoldcv_with_pr(kfold, 
                                     labels, 
                                     create_predictor(E))
    meth   = "F1"
    for i in range(len(metric)):
        print(f"Fold {i + 1} {meth}: {metric[i]}")
    results["F1"]     = metric
    print(f"Mean {meth}: {np.mean(metric)}: Std: {np.std(metric)}")

    metric = scoring.kfoldcv(kfold,
                             labels,
                             create_predictor(E, confidence = False))
    meth   = "ACC"
    for i in range(len(metric)):
        print(f"Fold {i + 1} {meth}: {metric[i]}")
    results["ACC"] = metric
    print(f"Mean {meth}: {np.mean(metric)}: Std: {np.std(metric)}")
    
    df = pd.DataFrame(results)
    print(f"Saving...")
    df.to_csv(args.output)
if __name__ == "__main__":
    main()
    
