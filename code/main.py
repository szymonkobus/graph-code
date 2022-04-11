import argparse
import sys

import numpy as np
import torch
import yaml

from graph import get_graph
from lossless import lossless_code_NP


def run(graph):
    print(graph)
    print(graph.adj)

if __name__=='__main__':
    config_file = sys.argv[1]
    with open(config_file) as file:
        config_dic = yaml.safe_load(file)
    conf = argparse.Namespace(**config_dic)
    
    graph = get_graph(conf)
    run(graph)
