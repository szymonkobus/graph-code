import argparse
import sys

import yaml

from bounds import (comm_bound, dist_comm_bound, dist_comm_bound_uniform,
                    distance_bound)
from comm import get_comm
from graph import get_graph, get_start
from lossless import lossless_code
from lossy import (expected_depth, junction_code, node_code_perf,
                   static_path_code_perf)
from prob import get_path_prob, get_prob


def run(graph, prob, start, comm, path_prob):
    # GRAPH
    print(f'Graph:\n{graph}\n')

    # BOUNDS
    L_d = distance_bound(graph, start, prob)
    L_c = comm_bound(comm, prob)
    L_cd = dist_comm_bound(graph, start, comm, prob)
    # L_cd_u = dist_comm_bound_uniform(graph, start, comm)
    print(f'distance bound      = {L_d:.5f}')
    print(f'comm bound          = {L_c:.5f}')
    print(f'comm distance bound = {L_cd:.5f}')
    print()
    # print('comm distance bound unifrom = {}'.format(L_cd_u))

    # CODING:
    # 1. node coding
    assert comm.period==1
    node_perf = node_code_perf(graph, prob, start, comm.n_symbols)
    print(f'node coding perf:         {node_perf:.5f}')
    
    # 2. static path coding
    paths = lossless_code(graph, start)
    node_paths = path_prob(paths, len(graph), prob)
    static_path_perf = static_path_code_perf(paths, prob, node_paths, 
                                             comm.n_symbols)
    print(f'static path coding perf:  {static_path_perf:.5f}')
    
    # 3. dynamic path coding
    tree = junction_code(paths, prob, node_paths, comm.n_symbols)
    dynamic_path_perf = expected_depth(tree, prob)
    print(f'dynamic path coding perf: {dynamic_path_perf:.5f}')
    
    print()
    print('n.o. shortest paths: {}'.format(len(paths)))
    # print(tree.draw())


if __name__ == '__main__':
    config_file = sys.argv[1]
    with open(config_file) as file:
        config_dic = yaml.safe_load(file)
    conf = argparse.Namespace(**config_dic)
    print('CONF : {}'.format(conf))

    graph = get_graph(conf)
    start = get_start(conf, len(graph))
    prob = get_prob(conf, len(graph))
    path_prob = get_path_prob(conf)
    comm = get_comm(conf)
    run(graph, prob, start, comm, path_prob)
