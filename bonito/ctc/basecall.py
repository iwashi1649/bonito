"""
Bonito basecall
"""

import os

import torch
import numpy as np
from functools import partial

from bonito.multiprocessing import process_map
from bonito.util import mean_qscore_from_qstring
from bonito.util import chunk, stitch, batchify, unbatchify, permute


def basecall(
    model, reads, beamsize=5, chunksize=0, overlap=0, batchsize=1, qscores=False,
    reverse=None, scores_out_dir=None, scores_out_format="npy", scores_only=False,
):
    """
    Basecalls a set of reads.
    """
    chunks = (
        (read, chunk(torch.tensor(read.signal), chunksize, overlap)) for read in reads
    )
    scores = unbatchify(
        (k, compute_scores(model, v)) for k, v in batchify(chunks, batchsize)
    )
    def stitch_and_export(scores_iter):
        for index, (read, values) in enumerate(scores_iter):
            stitched = stitch(values, chunksize, overlap, len(read.signal), model.stride)
            if scores_out_dir is not None:
                os.makedirs(scores_out_dir, exist_ok=True)
                name = getattr(read, 'name', None) or getattr(read, 'read_id', None) or str(index)
                name = str(name).split('!')[1] if '!' in str(name) else str(name)
                name = ''.join(char if (char.isalnum() or char in ('-', '_')) else '_' for char in name)
                data = stitched.cpu().numpy() if isinstance(stitched, torch.Tensor) else np.asarray(stitched)
                if data.ndim == 1:
                    data = np.atleast_2d(data).T
                shifted = data - np.max(data, axis=1, keepdims=True)
                probabilities = np.exp(shifted)
                data = probabilities / np.sum(probabilities, axis=1, keepdims=True)
                if scores_out_format == "csv":
                    np.savetxt(os.path.join(scores_out_dir, f"{name}.csv"), data, delimiter=",", fmt="%.6e")
                else:
                    np.save(os.path.join(scores_out_dir, f"{name}.npy"), data)
            yield read, {'scores': stitched}

    scores = stitch_and_export(scores)
    if scores_only:
        for _ in scores:
            pass
        return iter(())
    decoder = partial(decode, decode=model.decode, beamsize=beamsize, qscores=qscores, stride=model.stride)
    basecalls = process_map(decoder, scores, n_proc=4)
    return basecalls


def compute_scores(model, batch):
    """
    Compute scores for model.
    """
    with torch.no_grad():
        device = next(model.parameters()).device
        chunks = batch.to(torch.half).to(device)
        probs = permute(model(chunks), 'TNC', 'NTC')
    return probs.cpu().to(torch.float32)


def decode(scores, decode, beamsize=5, qscores=False, stride=1):
    """
    Convert the network scores into a sequence.
    """
    # do a greedy decode to get a sensible qstring to compute the mean qscore from
    seq, path = decode(scores['scores'], beamsize=1, qscores=True, return_path=True)
    seq, qstring = seq[:len(path)], seq[len(path):]
    mean_qscore = mean_qscore_from_qstring(qstring)

    # beam search will produce a better sequence but doesn't produce a sensible qstring/path
    if not (qscores or beamsize == 1):
        try:
            seq = decode(scores['scores'], beamsize=beamsize)
            path = None
            qstring = '*'
        except:
            pass

    return {'sequence': seq, 'qstring': qstring, 'stride': stride, 'moves': path}
