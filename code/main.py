import argparse
import sys
from typing import Generator

import yaml
from torch import Tensor
from tqdm import tqdm

from bounds import comm_bound, dist_comm_bound, distance_bound
from comm import get_comm
from graph import Graph, get_graph
from lossless import Paths, lossless_code
from lossy import (expected_depth, junction_code, node_code_perf,
                   static_path_code_perf)
from prob import get_path_prob, get_prob
from writer import get_writer


def loop(conf, graph_writer) \
        -> Generator[tuple[Graph, Paths, int, Tensor], None, None]:
    graph_writer.group_index(conf.n_start)
    for graph_i in range(conf.n_graph):
        if conf.load_graph and graph_i < len(graph_writer.index):
            graph, tree_writer = graph_writer.load(graph_i)
        else:
            graph = get_graph(conf)
            tree_writer = graph_writer.create(graph_i)
        if conf.save_graph:
            _ = graph.adj_sparse
            graph_writer.save(graph_i, graph)
        
        tree_writer.make_index_complement(len(graph))
        if type(conf.start) is int:
            tree_writer.force_start(conf.start)
        for start_i in range(conf.n_start):            
            if conf.load_paths and start_i < len(tree_writer.index):
                paths, start = tree_writer.load(start_i)
            else:
                start = tree_writer.index_complement.pop()
                tree_writer.create(start_i, start)
                paths = lossless_code(graph, start)
            if conf.save_paths:
                tree_writer.save(start_i, paths)
       
            for _ in range(conf.n_prob):
                prob = get_prob(conf, len(graph))
                yield graph, paths, start, prob


def run(conf, comm, path_prob, graph_writer):
    tot = conf.n_graph * conf.n_start * conf.n_prob
    for graph, paths, start, prob in tqdm(loop(conf, graph_writer), total=tot,
                                          disable=(conf.verbose), width=79):
        L_d = distance_bound(graph, start, prob)
        L_c = comm_bound(comm, prob)
        L_cd = dist_comm_bound(graph, start, comm, prob)
        # L_cd_u = dist_comm_bound_uniform(graph, start, comm)
        # 1. node coding
        assert comm.period==1
        node_perf = node_code_perf(graph, prob, start, comm.n_symbols)
        # 2. static path coding
        node_paths = path_prob(paths, len(graph), prob)
        static_path_perf = static_path_code_perf(paths, prob, node_paths, 
                                                 comm.n_symbols)
        # 3. dynamic path coding
        tree = junction_code(paths, prob, node_paths, comm.n_symbols)
        dynamic_path_perf = expected_depth(tree, prob)

        if conf.verbose:
            # print(f'Graph:\n{graph}\n{start}')
            print(f'distance bound:           {L_d:.5f}')
            print(f'comm bound:               {L_c:.5f}')
            print(f'comm distance bound:      {L_cd:.5f}')
            # print(f'comm distance bound unifrom = {L_cd_u:.5f}')
            print(f'node coding perf:         {node_perf:.5f}')
            print(f'static path coding perf:  {static_path_perf:.5f}')
            print(f'dynamic path coding perf: {dynamic_path_perf:.5f}')    
            print()
        
        if conf.table:
            print('{},{},{},{},{},{}'.format(
                L_d, L_c, L_cd, node_perf, static_path_perf, dynamic_path_perf))


def config_check(conf):
    assert not (conf.n_graph!=1 and conf.graph_type=='grid')
    assert not (conf.n_start!=1 and type(conf.start) is int)
    assert not (conf.n_prob!=1 and not conf.prob_permute)
    # TODO: add assert: using more n_start then there are points

if __name__ == '__main__':
    config_file = sys.argv[1]
    with open(config_file) as file:
        config_dic = yaml.safe_load(file)
    conf = argparse.Namespace(**config_dic)
    print('# CONF : {}'.format(conf))
    if conf.table:
        print('L_d,L_c,L_cd,P_n,P_SP,P_DP')

    config_check(conf)
    path_prob = get_path_prob(conf)
    comm = get_comm(conf)
    graph_writer = get_writer(conf.save_path, conf)
    run(conf, comm, path_prob, graph_writer)
