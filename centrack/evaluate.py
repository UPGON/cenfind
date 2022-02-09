import numpy as np
from numpy.random import default_rng
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

rng = default_rng(1993)


def generate_synthetic_data(height=512, size=200, has_daughter=.8):
    # Generate ground truth objects (true positives)
    foci = rng.integers(0, height, size=(size, 2))
    daughter_n = int(has_daughter * size)
    offset = rng.integers(-4, 4, size=(daughter_n, 2))

    daughters = rng.choice(foci, daughter_n, replace=False) + offset
    foci = np.concatenate([foci, daughters])

    return foci


def generate_predictions(height, foci, fn_rate=.1, fp_rate=.2, random=False):
    size = len(foci)

    fp_n = int(fp_rate * size)
    fn_n = int(fn_rate * size)

    if random:
        predictions = rng.integers(0, height, size=(50, 2))
        return predictions
    else:
        # Simulate the predictions and delete some objects (the false negatives)
        predictions = foci.copy()
        predictions = rng.choice(predictions, size - fn_n, replace=False)
        fps = rng.integers(0, height, size=(fp_n, 2))
        predictions = np.concatenate([predictions, fps], axis=0)

        return predictions


def compute_metrics(positions, predictions, offset_max):
    # Assign the predictions to the ground truth using the Hungarian algorithm.
    cost_matrix = cdist(positions, predictions)
    agents, tasks = linear_sum_assignment(cost_matrix, maximize=False)

    tps = set()
    fps = set()
    fns = set()

    for agent, task in zip(agents, tasks):
        if cost_matrix[agent, task] > offset_max:
            fps.add(task)
            fns.add(agent)
        else:
            tps.add(task)

    pos_idx = set(range(len(positions)))
    new_fns = pos_idx.difference(set(agents))
    fns = fns.union(new_fns)

    pred_idx = set(range(len(predictions)))
    new_fps = pred_idx.difference(set(tasks))
    fps = fps.union(new_fps)

    return {'fp': fps,
            'fn': fns,
            'tp': tps,
            }
