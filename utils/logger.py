""" Logger configuration """
import sys
import logging
from logging.handlers import RotatingFileHandler


logger = logging.getLogger("Image classification")
logger.setLevel(logging.INFO)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(filename)s] %(message)s: ",
    datefmt="%d.%m.%Y %H:%M:%S",
    handlers=(
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            "Image_classification.log",
            mode="a",
            maxBytes=200 * 1024 * 1024,
            backupCount=2,
            encoding="UTF-8",
        ),
    ),
)