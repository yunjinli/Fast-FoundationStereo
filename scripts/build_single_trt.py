#!/usr/bin/env python3
"""Build a TensorRT engine from the single-model ONNX export without trtexec."""

import argparse
import logging
from pathlib import Path

import tensorrt as trt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', required=True, type=Path)
    parser.add_argument('--engine', required=True, type=Path)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--workspace-gib', type=float, default=8.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if not args.onnx.is_file():
        raise FileNotFoundError(f'ONNX model not found: {args.onnx}')

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    onnx_parser = trt.OnnxParser(network, logger)

    logging.info('Parsing %s', args.onnx)
    if not onnx_parser.parse_from_file(str(args.onnx)):
        errors = '\n'.join(str(onnx_parser.get_error(i))
                           for i in range(onnx_parser.num_errors))
        raise RuntimeError(f'TensorRT could not parse the ONNX model:\n{errors}')

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(args.workspace_gib * (1 << 30)))
    if args.fp16:
        if not builder.platform_has_fast_fp16:
            raise RuntimeError('This GPU does not report fast FP16 support')
        config.set_flag(trt.BuilderFlag.FP16)

    logging.info('Building %s engine; this can take several minutes',
                 'FP16' if args.fp16 else 'FP32')
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TensorRT engine build failed; inspect the log above')

    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(serialized)
    logging.info('Wrote %s (%.1f MiB)', args.engine,
                 args.engine.stat().st_size / (1 << 20))


if __name__ == '__main__':
    main()
