import argparse
import sys

import numpy as np
import torch
import yaml

from bounds import comm_bound, dist_comm_bound, distance_bound
from comm import get_comm
from graph import create_grid, get_graph
from lossless import lossless_code


def run(graph, start, comm):
    # print(graph)
    # print(graph.adj)
    # paths = lossless_code(graph, 20)
    # print(paths)
    # print(len(paths))

    L_d = distance_bound(graph, start)
    print('distance bound      = {}'.format(L_d))
    L_c = comm_bound(comm, len(graph))
    print('comm bound          = {}'.format(L_c))
    L_cd = dist_comm_bound(graph, start, comm)
    print('comm distance bound = {}'.format(L_cd))



if __name__=='__main__':
    config_file = sys.argv[1]
    with open(config_file) as file:
        config_dic = yaml.safe_load(file)
    conf = argparse.Namespace(**config_dic)
    print('CONF : {}'.format(conf))
    
    graph = get_graph(conf)
    comm = get_comm(conf)
    run(graph, conf.start, comm)
