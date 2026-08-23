"""
Bonito CRF basecalling
"""

import os
import sys
import json
from time import perf_counter

import torch
import numpy as np
from koi.decode import beam_search, to_str

from bonito.multiprocessing import thread_iter
from bonito.util import chunk, stitch, batchify, unbatchify


def _synchronize(device):
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


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


def compute_transition_scores(model, batch, blank_score=2.0, reverse=False, timing=None,
                              keep_on_device=False):
    """Compute CRF transitions for ``--scores-only`` without decoding reads."""
    with torch.inference_mode():
        device = next(model.parameters()).device
        model_input = batch.to(torch.float16).to(device)
        _synchronize(device)
        started = perf_counter()
        scores = model(model_input)
        if reverse:
            scores = model.seqdist.reverse_complement(scores)
        _synchronize(device)
        model_forward_seconds = perf_counter() - started

        started = perf_counter()
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
        _synchronize(device)
        transition_probability_seconds = perf_counter() - started

        started = perf_counter()
        tracebacks = transitions.to(torch.float32).transpose(0, 1)
        initial_state = initial_state.to(torch.float32).unsqueeze(1)
        if not keep_on_device:
            tracebacks = tracebacks.cpu()
            initial_state = initial_state.cpu()
            _synchronize(device)
        tensor_prepare_seconds = perf_counter() - started
        if timing is not None:
            timing["batches"].append({
                "batch_size": int(batch.shape[0]),
                "model_forward_seconds": model_forward_seconds,
                "transition_probability_seconds": transition_probability_seconds,
                "cpu_transfer_array_seconds": (
                    0.0 if keep_on_device else tensor_prepare_seconds),
                "device_tensor_prepare_seconds": (
                    tensor_prepare_seconds if keep_on_device else 0.0),
            })
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
             reverse=False, rna=False, scores_out_dir=None, scores_out_format="npy", scores_only=False,
             blank_score=2.0, scores_timing=None, scores_decoder=None,
             scores_decoder_out_dir=None, scores_consumer=None):
    """
    Basecalls a set of reads.
    """
    chunks = thread_iter(
        ((read, 0, read.signal.shape[-1]), chunk(torch.from_numpy(read.signal), chunksize, overlap))
        for read in reads
    )

    batches = thread_iter(batchify(chunks, batchsize=batchsize))

    score_function = compute_transition_scores if scores_only else compute_scores
    scores = thread_iter(
        (read, score_function(model, batch, reverse=reverse, blank_score=blank_score,
                              **({"timing": scores_timing,
                                  "keep_on_device": scores_consumer is not None}
                                 if scores_only else {})))
        for read, batch in batches
    )

    results = thread_iter(
        (read, stitch_results(scores, end - start, chunksize, overlap, model.stride, reverse))
        for ((read, start, end), scores) in unbatchify(scores)
    )

    def export_scores(results_iter):
        try:
            for read, attrs in results_iter:
                name = getattr(read, 'name', None) or getattr(read, 'read_id', None) or str(id(read))
                name = str(name).split('!')[1] if '!' in str(name) else str(name)
                name = ''.join(char if (char.isalnum() or char in ('-', '_')) else '_' for char in name)
                started = perf_counter()
                if scores_consumer is not None:
                    scores_consumer(name, attrs['scores'], attrs['initial_state'])
                    array_seconds = 0.0
                    write_seconds = 0.0
                else:
                    scores_array = attrs['scores'].numpy()
                    initial_array = attrs['initial_state'].numpy()
                    array_seconds = perf_counter() - started
                    write_seconds = 0.0
                    if scores_out_dir is not None:
                        os.makedirs(scores_out_dir, exist_ok=True)
                        started = perf_counter()
                        save_scores_as_npy(scores_out_dir, name, scores_array)
                        save_initial_state_as_npy(scores_out_dir, name, initial_array)
                        write_seconds = perf_counter() - started
                    if scores_decoder is not None:
                        started = perf_counter()
                        decoded = scores_decoder(scores_array, initial_array)
                        decode_seconds = perf_counter() - started
                        os.makedirs(scores_decoder_out_dir, exist_ok=True)
                        with open(os.path.join(scores_decoder_out_dir, f"{name}.json"), "w", encoding="utf-8") as handle:
                            json.dump(decoded, handle, indent=2)
                            handle.write("\n")
                        if scores_timing is not None:
                            scores_timing.setdefault("direct_decodes", []).append({
                                "read_id": name, "decode_seconds": decode_seconds,
                            })
                if scores_timing is not None:
                    scores_timing["reads"].append({
                        "read_id": name,
                        "frames": int(attrs['scores'].shape[0]),
                        "array_view_seconds": array_seconds,
                        "npy_serialization_write_seconds": write_seconds,
                    })
                yield read, attrs
        finally:
            if scores_consumer is not None:
                finalize = getattr(scores_consumer, "finalize", None)
                if finalize is not None:
                    finalize()

    if scores_only:
        outputs = sum(item is not None for item in (
            scores_out_dir, scores_decoder, scores_consumer))
        if outputs == 0:
            raise ValueError("scores_out_dir, scores_decoder, or scores_consumer is required when scores_only is enabled")
        if scores_consumer is not None and outputs != 1:
            raise ValueError("scores_consumer cannot be combined with score export or decoder")
        if scores_out_dir is not None:
            sys.stderr.write(f"Scores will be saved to {scores_out_dir} in {scores_out_format} format\n")
        if scores_decoder is not None:
            if scores_decoder_out_dir is None:
                raise ValueError("scores_decoder_out_dir is required with scores_decoder")
            sys.stderr.write(f"Direct decoder results will be saved to {scores_decoder_out_dir}\n")
        if scores_consumer is not None:
            sys.stderr.write("CRF scores will be consumed on their model device\n")
        results = export_scores(results)
        return thread_iter((read, None) for read, _ in results)

    return thread_iter(
        (read, fmt(model.stride, attrs, rna))
        for read, attrs in results
    )
