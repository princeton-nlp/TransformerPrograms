import csv
import json
import logging
from pathlib import Path
import sys


def get_handler(fn=None, stream=None):
    formatter = logging.Formatter(
        fmt="%(levelname).1s %(asctime)s %(name)s:%(lineno)d: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    h = (
        logging.FileHandler(fn, mode="w")
        if fn
        else logging.StreamHandler(stream=stream)
    )
    h.setLevel(logging.INFO)
    h.setFormatter(formatter)
    return h


def get_resume_name(output_dir, s="output.log"):
    output_logs = list(Path(output_dir).glob(f"{s}*"))
    if len(output_logs) == 0:
        return Path(output_dir) / s
    idx = 1
    for p in output_logs:
        if str(p)[-1].isdigit():
            p_idx = int(str(p).split(".")[-1])
            idx = max(idx, p_idx + 1)
    return Path(output_dir) / f"{s}.{idx}"


def initialize(output_dir, resume=False):
    fn = (
        get_resume_name(output_dir)
        if resume
        else Path(output_dir) / "output.log"
    )
    if not fn.parent.exists():
        fn.parent.mkdir(parents=True)
    handlers = (
        get_handler(stream=sys.stdout),
        get_handler(fn=fn),
    )
    logging.basicConfig(handlers=handlers, force=True, level=logging.INFO)


def get_logger(name):
    logging.basicConfig(handlers=[get_handler(stream=None)], level=logging.INFO)
    return logging.getLogger(name)
