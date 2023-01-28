import logging
from datetime import datetime
from pathlib import Path

# TODO: Implement logger objects
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
start_stamp = datetime.now()
log_file = f'{start_stamp.strftime("%Y%m%d_%H:%M:%S")}_train.log'
fh = logging.FileHandler(Path("./logs") / log_file)
fh.setFormatter(formatter)
logger.addHandler(fh)
