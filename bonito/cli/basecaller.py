"""
Bonito Basecaller
"""

import os
import sys
import inspect
import json
import numpy as np
from tqdm import tqdm
from time import perf_counter
from functools import partial
from datetime import timedelta
from itertools import islice as take
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bonito.nn import fuse_bn_
from bonito.aligner import align_map, Aligner
from bonito.reader import read_chunks, Reader
from bonito.io import CTCWriter, Writer, biofmt
from bonito.cli.download import Downloader, models, __models_dir__
from bonito.multiprocessing import process_cancel, process_itemmap
from bonito.util import column_to_set, load_symbol, load_model, init, tqdm_environ


def main(args):

    init(args.seed, args.device)

    try:
        reader = Reader(args.reads_directory, args.recursive)
        sys.stderr.write("> reading %s\n" % reader.fmt)
    except FileNotFoundError:
        sys.stderr.write("> error: no suitable files found in %s\n" % args.reads_directory)
        exit(1)

    fmt = biofmt(aligned=args.reference is not None)

    if args.reference and args.reference.endswith(".mmi") and fmt.name == "cram":
        sys.stderr.write("> error: reference cannot be a .mmi when outputting cram\n")
        exit(1)
    elif args.reference and fmt.name == "fastq":
        sys.stderr.write(f"> warning: did you really want {fmt.aligned} {fmt.name}?\n")
    else:
        sys.stderr.write(f"> outputting {fmt.aligned} {fmt.name}\n")

    if args.model_directory in models and not (__models_dir__ / args.model_directory).exists():
        sys.stderr.write("> downloading model\n")
        Downloader(__models_dir__).download(args.model_directory)

    sys.stderr.write(f"> loading model {args.model_directory}\n")
    try:
        model = load_model(
            args.model_directory,
            args.device,
            weights=args.weights if args.weights > 0 else None,
            chunksize=args.chunksize,
            overlap=args.overlap,
            batchsize=args.batchsize,
            quantize=args.quantize,
            use_koi=True,
        )
        model = model.apply(fuse_bn_)
    except FileNotFoundError:
        sys.stderr.write(f"> error: failed to load {args.model_directory}\n")
        sys.stderr.write(f"> available models:\n")
        for model in sorted(models): sys.stderr.write(f" - {model}\n")
        exit(1)

    if args.verbose:
        sys.stderr.write(f"> model basecaller params: {model.config['basecaller']}\n")

    basecall = load_symbol(args.model_directory, "basecall")

    if args.reference:
        sys.stderr.write("> loading reference\n")
        aligner = Aligner(args.reference, preset=args.mm2_preset)
        if not aligner:
            sys.stderr.write("> failed to load/build index\n")
            exit(1)
    else:
        aligner = None

    if args.save_ctc and not args.reference:
        sys.stderr.write("> a reference is needed to output ctc training data\n")
        exit(1)

    calibration_consumer = None
    calibration_args = (
        args.calibration_reference_fasta,
        args.calibration_manifest,
        args.calibration_summary,
    )
    if any(value is not None for value in calibration_args):
        if not all(value is not None for value in calibration_args):
            sys.stderr.write(
                "> --calibration-reference-fasta, --calibration-manifest, and "
                "--calibration-summary are required together\n")
            exit(1)
        if not args.scores_only:
            sys.stderr.write("> calibration statistics require --scores-only\n")
            exit(1)
        from nanopore_dna_storage.decoding.bonito_calibration_consumer import (
            BonitoCalibrationConsumer,
        )
        calibration_consumer = BonitoCalibrationConsumer(
            args.calibration_reference_fasta,
            args.calibration_manifest,
            args.calibration_summary,
            batch_size=args.calibration_batch_size,
            trim_bases=args.calibration_trim_bases,
            checkpoint_every_reads=args.calibration_checkpoint_every,
            resume=args.calibration_resume,
            phase=args.calibration_phase,
            learning_rate=args.calibration_learning_rate,
            shrinkage_support=args.calibration_shrinkage_support,
            optimizer_interval_batches=args.calibration_optimizer_interval_batches,
        )

    if fmt.name != 'fastq':
        groups, num_reads = reader.get_read_groups(
            args.reads_directory, args.model_directory,
            n_proc=8, recursive=args.recursive,
            read_ids=column_to_set(args.read_ids), skip=args.skip,
            cancel=process_cancel()
        )
    else:
        groups = []
        num_reads = None

    reads = reader.get_reads(
        args.reads_directory, n_proc=8, recursive=args.recursive,
        read_ids=column_to_set(args.read_ids), skip=args.skip,
        do_trim=not args.no_trim,
        scaling_strategy=model.config.get("scaling"),
        norm_params=(model.config.get("standardisation")
                     if (model.config.get("scaling") and
                         model.config.get("scaling").get("strategy") == "pa")
                     else model.config.get("normalisation")
                     ),
        cancel=process_cancel()
    )

    if args.verbose:
        sys.stderr.write(f"> read scaling: {model.config.get('scaling')}\n")
    
    if args.max_reads:
        reads = take(reads, args.max_reads)
        if num_reads is not None:
            num_reads = min(num_reads, args.max_reads)

    if args.save_ctc:
        reads = (
            chunk for read in reads
            for chunk in read_chunks(
                read,
                chunksize=model.config["basecaller"]["chunksize"],
                overlap=model.config["basecaller"]["overlap"]
            )
        )
        ResultsWriter = CTCWriter
    else:
        ResultsWriter = Writer

    basecall_kwargs = dict(
        batchsize=model.config["basecaller"]["batchsize"],
        chunksize=model.config["basecaller"]["chunksize"],
        overlap=model.config["basecaller"]["overlap"],
        reverse=args.revcomp,
        rna=args.rna,
        scores_only=args.scores_only,
        blank_score=args.blank_score,
    )
    score_timing = None
    if args.scores_only:
        basecall_kwargs["scores_out_format"] = args.scores_format
        if args.scores_dir is not None:
            basecall_kwargs["scores_out_dir"] = args.scores_dir
        if args.scores_timing_json is not None:
            score_timing = {"schema_version": 1, "batches": [], "reads": []}
            basecall_kwargs["scores_timing"] = score_timing
        if args.hedges_crf_direct_results_dir is not None:
            from nanopore_dna_storage.decoding.hedges_crf_cpp_binding import load_binding
            state_calibration = None
            if args.hedges_crf_calibration_matrix is not None:
                import json
                from pathlib import Path
                from nanopore_dna_storage.decoding.crf_calibration import load_calibration_payload
                payload = json.loads(Path(args.hedges_crf_calibration_matrix).read_text(encoding="utf-8"))
                state_calibration, _ = load_calibration_payload(payload)
            binding = load_binding()
            basecall_kwargs["scores_decoder"] = lambda scores, initial: binding.decode_probabilities(
                scores, initial, "TCGAAGTCAGCGTGTATTGTATG", "AGTAGTGAGTGCGATTAAGCGTGTT",
                coderatecode=args.hedges_crf_coderatecode,
                state_calibration_matrices=state_calibration)
            basecall_kwargs["scores_decoder_out_dir"] = args.hedges_crf_direct_results_dir
        if calibration_consumer is not None:
            basecall_kwargs["scores_consumer"] = calibration_consumer
    accepted = set(inspect.signature(basecall).parameters)
    score_export_started = perf_counter()
    results = basecall(model, reads, **{key: value for key, value in basecall_kwargs.items() if key in accepted})

    if args.scores_only and "scores_only" in accepted:
        for _ in tqdm(results, desc="> exporting scores", unit=" reads", leave=False,
                      total=num_reads, smoothing=0, ascii=True, ncols=100,
                      **tqdm_environ()):
            pass
        score_export_wall = perf_counter() - score_export_started
        if score_timing is not None:
            batch_fields = ("model_forward_seconds", "transition_probability_seconds",
                            "cpu_transfer_array_seconds", "device_tensor_prepare_seconds")
            totals = {field: sum(row[field] for row in score_timing["batches"])
                      for field in batch_fields}
            totals["array_view_seconds"] = sum(row["array_view_seconds"] for row in score_timing["reads"])
            totals["npy_serialization_write_seconds"] = sum(
                row["npy_serialization_write_seconds"] for row in score_timing["reads"])
            totals["direct_decode_seconds"] = sum(
                row["decode_seconds"] for row in score_timing.get("direct_decodes", []))
            totals["score_export_wall_seconds"] = score_export_wall
            measured = sum(value for key, value in totals.items() if key != "score_export_wall_seconds")
            totals["stage_sum_seconds"] = measured
            totals["stage_sum_minus_wall_seconds"] = measured - score_export_wall
            score_timing["aggregate"] = {
                "batch_count": len(score_timing["batches"]),
                "read_count": len(score_timing["reads"]),
                **totals,
                "note": "stage sums may overlap because Bonito pipelines work with thread_iter",
            }
            timing_path = os.path.abspath(args.scores_timing_json)
            os.makedirs(os.path.dirname(timing_path), exist_ok=True)
            with open(timing_path, "w", encoding="utf-8") as handle:
                json.dump(score_timing, handle, indent=2)
                handle.write("\n")
        sys.stderr.write("> exported scores only (no decode / no fastq)\n")
        return

    if aligner:
        results = align_map(aligner, results, n_thread=args.alignment_threads)

    writer_kwargs = {'aligner': aligner,
                     'group_key': args.model_directory,
                     'ref_fn': args.reference,
                     'groups': groups,
                     'min_qscore': args.min_qscore}
    if args.save_ctc:
        writer_kwargs['rna'] = args.rna
        writer_kwargs['min_accuracy'] = args.min_accuracy_save_ctc
        
    writer = ResultsWriter(
        fmt.mode, tqdm(results, desc="> calling", unit=" reads", leave=False,
                       total=num_reads, smoothing=0, ascii=True, ncols=100,
                       **tqdm_environ()),
        **writer_kwargs)

    t0 = perf_counter()
    writer.start()
    writer.join()
    duration = perf_counter() - t0
    num_samples = sum(num_samples for read_id, num_samples in writer.log)

    sys.stderr.write("> completed reads: %s\n" % len(writer.log))
    sys.stderr.write("> duration: %s\n" % timedelta(seconds=np.round(duration)))
    sys.stderr.write("> samples per second %.1E\n" % (num_samples / duration))
    sys.stderr.write("> done\n")


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("model_directory")
    parser.add_argument("reads_directory")
    parser.add_argument("--reference")
    parser.add_argument("--read-ids")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--weights", default=0, type=int)
    parser.add_argument("--skip", action="store_true", default=False)
    parser.add_argument("--no-trim", action="store_true", default=False)
    parser.add_argument("--save-ctc", action="store_true", default=False)
    parser.add_argument("--revcomp", action="store_true", default=False)
    parser.add_argument("--rna", action="store_true", default=False)
    parser.add_argument("--recursive", action="store_true", default=False)
    parser.add_argument("--scores-only", action="store_true", default=False,
                        help="Export per-read scores and skip decoding/FASTQ output")
    parser.add_argument("--scores-dir", default=None,
                        help="Directory for per-read score output")
    parser.add_argument("--scores-format", choices=["npy", "csv"], default="npy",
                        help="Per-read score output format")
    parser.add_argument("--scores-timing-json", default=None,
                        help="Write detailed score-export timing as JSON")
    parser.add_argument("--hedges-crf-direct-results-dir", default=None,
                        help="Decode CRF scores in memory with the experimental C++ HEDGES binding")
    parser.add_argument("--hedges-crf-calibration-matrix", default=None,
                        help="State-specific 5x5 calibration artifact for direct CRF-HEDGES decode")
    parser.add_argument("--hedges-crf-coderatecode", type=int, default=3,
                        help="HEDGES coderatecode for the experimental in-memory decoder")
    parser.add_argument("--calibration-reference-fasta", default=None,
                        help="Known per-read references for on-device calibration statistics")
    parser.add_argument("--calibration-manifest", default=None,
                        help="Fixed read_id/train-validation TSV for calibration statistics")
    parser.add_argument("--calibration-summary", default=None,
                        help="Atomic checkpoint and final compact calibration JSON")
    parser.add_argument("--calibration-batch-size", type=int, default=10,
                        help="Number of equal-reference-length reads per calibration DP batch")
    parser.add_argument("--calibration-trim-bases", type=int, default=10,
                        help="Reference bases excluded from each end")
    parser.add_argument("--calibration-checkpoint-every", type=int, default=1000,
                        help="Processed reads between atomic calibration checkpoints")
    parser.add_argument("--calibration-resume", action="store_true", default=False,
                        help="Restore compact aggregates and skip already processed read IDs")
    parser.add_argument("--calibration-phase",
                        choices=["statistics", "train", "validation"],
                        default="statistics",
                        help="Collect statistics, update matrices, or evaluate fixed matrices")
    parser.add_argument("--calibration-learning-rate", type=float, default=0.01,
                        help="Adam learning rate for global and state calibration matrices")
    parser.add_argument("--calibration-shrinkage-support", type=float, default=100.0,
                        help="State support scale for shrinkage toward the global matrix")
    parser.add_argument("--calibration-optimizer-interval-batches", type=int, default=1,
                        help="Posterior batches accumulated per Adam update")
    parser.add_argument("--blank-score", type=float, default=2.0,
                        help="CRF blank score used for normal decoding and --scores-only export")
    quant_parser = parser.add_mutually_exclusive_group(required=False)
    quant_parser.add_argument("--quantize", dest="quantize", action="store_true")
    quant_parser.add_argument("--no-quantize", dest="quantize", action="store_false")
    parser.set_defaults(quantize=None)
    parser.add_argument("--overlap", default=None, type=int)
    parser.add_argument("--chunksize", default=None, type=int)
    parser.add_argument("--batchsize", default=None, type=int)
    parser.add_argument("--max-reads", default=0, type=int)
    parser.add_argument("--min-qscore", default=0, type=int)
    parser.add_argument("--min-accuracy-save-ctc", default=0.99, type=float)
    parser.add_argument("--alignment-threads", default=8, type=int)
    parser.add_argument("--mm2-preset", default='lr:hq', type=str)
    parser.add_argument('-v', '--verbose', action='count', default=0)
    return parser
