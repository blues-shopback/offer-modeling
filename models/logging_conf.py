import os
import time
import logging

# TODO: make this more organized.
logger = None
log_file_path = None
log_filename = None
log_dir = None


def get_logger(working_dir=None):
    global logger, log_filename, log_file_path, log_dir
    if logger:
        return logger
    if working_dir:
        log_dir = os.path.join(working_dir, "log")
    else:
        log_dir = "/data/offer_modeling/log/"

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # Create file handler
    timestamp = str(int(time.time()))  # create time stamp for log file name.
    log_filename = "log_{}.txt".format(timestamp)
    log_file_path = os.path.join(log_dir, log_filename)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # Create logger
    logger = logging.getLogger('root')
    logger.setLevel(logging.DEBUG)

    # add handler to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info(f"Log file path: {log_file_path}")

    return logger
