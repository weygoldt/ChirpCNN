import logging


def make_logger(name: str):
    # create logger formats for file and terminal
    file_formatter = logging.Formatter(
        "[ %(levelname)s ] ~ %(asctime)s ~ %(module)s.%(funcName)s: %(message)s"
    )
    console_formatter = logging.Formatter(
        "[ %(levelname)s ] in %(module)s.%(funcName)s: %(message)s"
    )

    # create stream handler for terminal output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    # create stream handler for file output
    file_handler = logging.FileHandler("logfile.log")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)

    # create script specific logger
    logger = logging.getLogger(name)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    return logger


if __name__ == "__main__":
    # initiate logger
    mylogger = make_logger(__name__)

    # test logger levels
    mylogger.debug("This is for debugging!")
    mylogger.info("This is an info.")
    mylogger.warning("This is a warning.")
    mylogger.error("This is an error.")
    mylogger.critical("This is a critical error!")
