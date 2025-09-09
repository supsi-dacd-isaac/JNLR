# python
import logging
import sys

# ANSI colors
RESET = "\033[0m"
LIGHT_BLUE = "\033[94m"   # light blue for message (INFO and below)
VIOLET = "\033[95m"       # violet for WARNING
LIGHT_CYAN = "\033[96m"   # light cyan for datetime
RED = "\033[91m"          # red for ERROR/CRITICAL
WHITE = "\033[97m"        # white for DEBUG

class LevelColorFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.color_start_message = VIOLET
        record.color_start_time = LIGHT_BLUE
        record.color_start_level = LIGHT_CYAN

        return True

def configure_logging(name, level: int = logging.INFO) -> logging.Logger:
    handler = logging.StreamHandler(stream=sys.stderr)
    fmt = "%(color_start_time)s%(asctime)s%(color_start_level)s %(levelname)s %(name)s%(color_start_message)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    handler.addFilter(LevelColorFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    logger = logging.getLogger(name)
    return logger


