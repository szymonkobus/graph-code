from heapq import heapify, heappop, heappush
from typing import Callable, TypeVar


I = TypeVar('I')


def huffman(items: list[I], prob: list[float], join: Callable[..., I],
            degree: int = 2) -> I:
    heap = [(p, i, item) for i, (item, p) in enumerate(zip(items, prob))]
    if degree != 2:
        n_fill = (1-len(items)) % (degree-1)
        fill = [(0., i+len(items), None) for i in range(n_fill)]
        heap = fill + heap # type: ignore
    heapify(heap)
    while len(heap) >= degree:
        top = [heappop(heap) for _ in range(degree)]
        top_p, top_i, top_item = zip(*top)
        joined = (sum(top_p), min(top_i), join(*top_item))
        heappush(heap, joined)
    _, _, item = heap[0]
    return item
