import argparse
import sys

import numpy as np
import torch
import yaml

from bounds import (comm_bound, dist_comm_bound, dist_comm_bound_uniform,
                    distance_bound)
from comm import get_comm
from graph import create_grid, get_graph
from lossless import lossless_code
from prob import get_prob, get_path_prob


def run(graph, prob, start, comm, path_prob):
    print(graph)
    print(graph.adj)
    paths = lossless_code(graph, 20)
    print(paths)
    print(len(paths))
    # prob = torch.tensor([1/len(graph)]).expand(len(graph))
    L_d = distance_bound(graph, start, prob)
    # print('distance bound      = {}'.format(L_d))
    # L_c = comm_bound(comm, prob)
    # print('comm bound          = {}'.format(L_c))
    # L_cd = dist_comm_bound(graph, start, comm, prob)
    # print('comm distance bound = {}'.format(L_cd))
    # L_cd_u = dist_comm_bound_uniform(graph, start, comm)
    # print('comm distance bound unifrom = {}'.format(L_cd_u))


if __name__ == '__main__':
    config_file = sys.argv[1]
    with open(config_file) as file:
        config_dic = yaml.safe_load(file)
    conf = argparse.Namespace(**config_dic)
    print('CONF : {}'.format(conf))

    graph = get_graph(conf)
    prob = get_prob(conf, graph)
    path_prob = get_path_prob(conf)
    comm = get_comm(conf)
    run(graph, prob, conf.start, comm, path_prob)
