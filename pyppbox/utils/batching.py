# Copyright (c) 2026 UMONS-Numediart | AGPL-3.0 License
"""Small backend-free helpers for explicitly bounded inference batches."""
from itertools import islice
from operator import index


def _batch_size(value):
    try:
        result = index(value)
    except TypeError as error:
        raise ValueError('batch_size must be a positive integer.') from error
    if isinstance(value, bool) or result < 1:
        raise ValueError('batch_size must be a positive integer.')
    return result


def _batches(values, size):
    iterator = iter(values)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            return
        yield chunk
