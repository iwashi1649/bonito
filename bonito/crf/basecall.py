"""
Bonito CRF basecalling
"""

import os
import sys

import torch
import numpy as np
from koi.decode import beam_search, to_str

from bonito.multiprocessing import thread_iter
from bonito.util import chunk, stitch, batchify, unbatchify


def stitch_results(results, length, size, overlap, stride, reverse=False):
    """
    Stitch results together with a given overlap.
    """
    if isinstance(results, dict):
        return {
            k: (v[0] if k == 'initial_state' else stitch_results(v, length, size, overlap, stride, reverse=reverse))
            for k, v in results.items()
        }
    if length < size:
        return results[0, :int(np.floor(length / stride))]
    return stitch(results, size, overlap, length, stride, reverse=reverse)


def compute_scores(model, batch, beam_width=32, beam_cut=100.0, scale=1.0, offset=0.0, blank_score=2.0, reverse=False):
    """
    Run the stock CRF beam decoder used for ordinary Bonito basecalling.
    """
    with torch.inference_mode():
        device = next(model.parameters()).device
        scores = model(batch.to(torch.float16).to(device))
        if reverse:
            scores = model.seqdist.reverse_complement(scores)
        with torch.cuda.device(scores.device):
            sequence, qstring, moves = beam_search(
                scores, beam_width=beam_width, beam_cut=beam_cut,
                scale=scale, offset=offset, blank_score=blank_score,
            )
        return {
            'moves': moves,
            'qstring': qstring,
            'sequence': sequence,
        }


def compute_transition_scores(model, batch, blank_score=2.0, reverse=False):
    """Compute CRF transitions for ``--scores-only`` without decoding reads."""
    with torch.inference_mode():
        device = next(model.parameters()).device
        scores = model(batch.to(torch.float16).to(device))
        if reverse:
            scores = model.seqdist.reverse_complement(scores)

        scores_pad = scores.permute(1, 0, 2)
        n_base = model.seqdist.n_base
        time_steps, batch_size, channels = scores_pad.shape
        scores_pad = torch.nn.functional.pad(
            scores_pad.view(time_steps, batch_size, channels // n_base, n_base),
            (1, 0, 0, 0, 0, 0, 0, 0),
            value=blank_score,
        ).view(time_steps, batch_size, -1)
        betas = model.seqdist.backward_scores(scores_pad.to(torch.float32))
        transitions, initial_state = model.seqdist.compute_transition_probs(scores_pad, betas)
        tracebacks = transitions.to(torch.float32).transpose(0, 1).cpu()
        initial_state = initial_state.to(torch.float32).unsqueeze(1).cpu()
        return {
            'scores': tracebacks,
            'initial_state': initial_state.squeeze(1),
        }


def fmt(stride, attrs, rna=False):
    fliprna = (lambda x:x[::-1]) if rna else (lambda x:x)
    return {
        'stride': stride,
        'moves': attrs['moves'].numpy(),
        'qstring': fliprna(to_str(attrs['qstring'])),
        'sequence': fliprna(to_str(attrs['sequence'])),
        'raw_scores': attrs.get('raw_scores'),
    }


def save_scores_as_npy(outdir, name, data):
    np.save(os.path.join(outdir, f"{name}_scores.npy"), data)


def save_initial_state_as_npy(outdir, name, data):
    np.save(os.path.join(outdir, f"{name}_initial_state.npy"), data)


def basecall(model, reads, chunksize=4000, overlap=100, batchsize=32,
             reverse=False, rna=False, scores_out_dir=None, scores_out_format="npy", scores_only=False):
    """
    Basecalls a set of reads.
    """
    chunks = thread_iter(
        ((read, 0, read.signal.shape[-1]), chunk(torch.from_numpy(read.signal), chunksize, overlap))
        for read in reads
    )

    batches = thread_iter(batchify(chunks, batchsize=batchsize))

    score_function = compute_transition_scores if scores_only else compute_scores
    scores = thread_iter((read, score_function(model, batch, reverse=reverse)) for read, batch in batches)

    results = thread_iter(
        (read, stitch_results(scores, end - start, chunksize, overlap, model.stride, reverse))
        for ((read, start, end), scores) in unbatchify(scores)
    )

    def export_scores(results_iter):
        for read, attrs in results_iter:
            if scores_out_dir is not None:
                os.makedirs(scores_out_dir, exist_ok=True)
                name = getattr(read, 'name', None) or getattr(read, 'read_id', None) or str(id(read))
                name = str(name).split('!')[1] if '!' in str(name) else str(name)
                name = ''.join(char if (char.isalnum() or char in ('-', '_')) else '_' for char in name)
                save_scores_as_npy(scores_out_dir, name, attrs['scores'].cpu().numpy())
                save_initial_state_as_npy(scores_out_dir, name, attrs['initial_state'].cpu().numpy())
            yield read, attrs

    if scores_only:
        if scores_out_dir is None:
            raise ValueError("scores_out_dir is required when scores_only is enabled")
        sys.stderr.write(f"Scores will be saved to {scores_out_dir} in {scores_out_format} format\n")
        results = export_scores(results)
        return thread_iter((read, None) for read, _ in results)

    return thread_iter(
        (read, fmt(model.stride, attrs, rna))
        for read, attrs in results
    )
