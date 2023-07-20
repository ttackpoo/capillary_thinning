__all__ = ["group_neighbors", "cycle_nocache"]


def group_neighbors(seq, n=2):
    """
    Group the neighboring *n* elements.

    Parameters
    ==========

    seq : iterable

    n : int
        Number of elements in each group

    Yields
    ======

    list of grouped elements

    Example
    =======

    >>> from mcplexpt.util.iterables import group_neighbors
    >>> seq = [1,2,3,4]
    >>> list(group_neighbors(seq, 2))
    [[1, 2], [2, 3], [3, 4]]
    >>> list(group_neighbors(seq, 3))
    [[1, 2, 3], [2, 3, 4]]

    """
    seq = list(seq)
    itertime = len(seq) - n + 1
    for i in range(itertime):
        result = []
        for j in range(n):
            result.append(seq[i + j])
        yield result


def cycle_nocache(func):
    """
    Cycle the generator without caching the elements.

    Parameters
    ==========

    func : function
        Nullary function which constructs and returns the generator.

    Examples
    ========

    >>> from mcplexpt.util import cycle_nocache
    >>> f = lambda : (i for i in range(3))
    >>> c = cycle_nocache(f)

    """
    gen = func()
    while True:
        try:
            ret = next(gen)
        except StopIteration:
            gen = func()
            ret = next(gen)
        yield ret
