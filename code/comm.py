from typing import Generator


class Comm:
    def __init__(self, n_symbols: int, period: int = -1) -> None:
        self.n_symbols = n_symbols
        self.period = period


def get_comm(conf) -> Comm:
    return Comm(conf.n_symbols, conf.period)


def comm_dist_iterator(comm: Comm) -> Generator[int, None, None]:
    yield 0
    width = comm.n_symbols
    depth = 1

    while True:
        for _ in range(width):
            yield depth
        depth += 1
        if comm.period != -1 and (depth + 1) % comm.period == 0:
            width *= comm.n_symbols
