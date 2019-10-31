import numpy as np 
import networkx as nx 


import os

def main():

    output_dir = os.path.join("datasets", "synthetic_scale_free")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    N = 1000

    for seed in range(100):
        g = nx.DiGraph(nx.scale_free_graph(N, seed=seed))
        print ("seed", seed, "number of nodes", len(g),
            "number of edges", len(g.edges))
        nx.set_edge_attributes(g, name="weight", values=1.)
        d = os.path.join(output_dir, "{:02d}".format(seed))
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
        nx.write_edgelist(g, os.path.join(d, "edgelist.tsv"), 
            delimiter="\t", data=["weight"])



if __name__ == "__main__":
    main()