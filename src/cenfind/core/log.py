import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_null_handler():
    null_handler = logging.NullHandler()
    null_handler.setFormatter(FORMATTER)
    return null_handler


def get_file_handler(file):
    file_handler = TimedRotatingFileHandler(file, when='midnight')
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_logger(logger_name, console=None, file=None):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # better to have too much log than not enough
    logger.addHandler(get_null_handler())
    if console:
        logger.addHandler(get_console_handler())
    if file:
        logger.addHandler(get_file_handler(file))

    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = False
    return logger


if __name__ == "__main__":
    print(Path(__file__).parents[3])
