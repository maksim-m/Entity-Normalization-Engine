import random
import string
from pathlib import Path
from typing import Union, List

random.seed(42)
OUTPUT_FILE = Path("processed/serial_number.txt")
N_SAMPLES = 26500


def random_string(length: Union[int, List[int]], chars: str = string.ascii_letters + string.digits) -> str:
    if isinstance(length, list):
        length = random.choice(length)
    return ''.join([random.choice(chars) for _ in range(length)])


def random_serial_number() -> str:
    """
    Generate one of the most common serial number types
    See https://support.google.com/merchants/answer/160161?hl=en
    """
    return random.choice([
        random_string(12, chars=string.digits),  # UPC / GTIN-12 / UPC-A
        random_string([13, 8, 14], chars=string.digits),  # GTIN-13
        random_string(13, chars=string.digits),  # ISBN-13
        random_string(list(range(4, 20))),  # MPN
    ])


serial_numbers = [random_serial_number() for _ in range(N_SAMPLES)]

with open(OUTPUT_FILE, 'w') as file:
    for sn in serial_numbers:
        file.write("%s\n" % sn)
