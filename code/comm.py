from typing import Generator


class Comm:
    def __init__(self, n_symbols: int, period: int = -1) -> None:
        self.n_symbols = n_symbols
        self.period = period

    def __getitem__(self, key: int) -> int:
        if self.period != -1 and key % self.period == 0:
            return self.n_symbols
        return 1


def get_comm(conf) -> Comm:
    return Comm(conf.n_symbols, conf.period)


def comm_dist_iterator(comm: Comm) -> Generator[int, None, None]:
    yield 0
    width = comm.n_symbols
    depth = 1

    while True:
        for _ in range(width):
            yield depth
        width *= comm[depth]
        depth += 1