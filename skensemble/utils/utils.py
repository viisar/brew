import numpy as np

from collections import Counter

def has_consensus(labels, options=None):
    if options is not None:
        labels = [l for l in labels if l in options]

    if len(labels) == 0:
        return False

    counter = Counter(labels)
    values = np.array(counter.values())
    return np.sum(np.max(values) == values) == 1

def get_consensus(labels, options=None):
    if options is not None:
        labels = [l for l in labels if l in options]

    if len(labels) == 0:
        return None

    counter = Counter(labels)
    values = np.array(counter.values())
    if np.sum(np.max(values) == values) == 1:
        return counter.most_common(1)[0][0]
    else:
        return None

def get_ties(labels, options=None):
    if options is not None:
        labels = [l for l in labels if l in options]

    if len(labels) == 0:
        return None

    counter = Counter(labels)
    values = np.array(counter.values())
    n_ties = np.sum(np.max(values) == values)

    ties = [item[0] for item in counter.most_common(n_ties)]
    return ties







