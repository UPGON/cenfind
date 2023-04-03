import argparse
import traceback
import sys
import importlib

from argparse import ArgumentDefaultsHelpFormatter
from cenfind.core.log import get_logger

commands = [
    "prepare",
    "squash",
    "score",
    "analyse",
    "evaluate",
    "predict",
    "train",
    "vignettes",
    "upload",
]

COMMANDS = [importlib.import_module("cenfind.cli." + c) for c in commands]


def add_command_default(parser):
    class default_command:
        def run(args):
            parser.print_help()
            return 2

    parser.set_defaults(__command__=default_command)


def add_command_subparsers(subparsers, commands, command_attribute="__command__"):
    for command in commands:
        subparser = command.register_parser(subparsers)

        if command_attribute:
            subparser.set_defaults(**{command_attribute: command})
            subparser.formatter_class = ArgumentDefaultsHelpFormatter

        if not subparser.description and command.__doc__:
            subparser.description = command.__doc__

        if not getattr(command, "run", None):
            add_command_default(subparser)


def make_parser():
    parser = argparse.ArgumentParser(
        prog="CENFIND",
        description="CENFIND: A deep-learning pipeline to score cells for centrioles",
    )

    add_command_default(parser)
    subparsers = parser.add_subparsers()
    add_command_subparsers(subparsers, COMMANDS)

    return parser


def run(argv):
    args = make_parser().parse_args(argv)

    logger = get_logger(__name__)

    try:
        return args.__command__.run(args)
    except RecursionError:
        logger.error("Maximum recursion depth reached.", exc_info=True)
        raise
    except FileNotFoundError:
        logger.error("File not found", exc_info=True)
        raise
