"""Perceptual hashing and Hamming-radius search for image audit."""
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy.fft import dctn


def perceptual_hash(image_or_path, hash_size=8, high_frequency_factor=4):
    """Return a 64-bit DCT perceptual hash as an integer."""
    side = hash_size * high_frequency_factor
    if isinstance(image_or_path, Image.Image):
        image = image_or_path
        close_image = False
    else:
        image = Image.open(image_or_path)
        close_image = True
    try:
        working_image = image
        if image.mode == "P" and "transparency" in image.info:
            working_image = image.convert("RGBA")
        pixels = np.asarray(
            working_image.convert("L").resize(
                (side, side), Image.Resampling.LANCZOS
            ),
            dtype=np.float32,
        )
    finally:
        if close_image:
            image.close()

    low_frequency = dctn(pixels, type=2, norm="ortho")[
        :hash_size, :hash_size
    ]
    flat = low_frequency.ravel()
    median = float(np.median(flat[1:]))
    bits = flat > median
    bits[0] = False  # Ignore the DC brightness component.
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hash_to_hex(value, hash_size=8):
    """Format a perceptual hash with stable zero padding."""
    return f"{int(value):0{hash_size * hash_size // 4}x}"


def hex_to_hash(value):
    return int(str(value), 16)


def hamming_distance(left, right):
    return (int(left) ^ int(right)).bit_count()


@dataclass
class _BKNode:
    value: int
    payloads: list = field(default_factory=list)
    children: dict = field(default_factory=dict)


class HammingBKTree:
    """BK-tree specialized for integer hashes and Hamming distance."""

    def __init__(self):
        self.root = None

    def add(self, value, payload):
        value = int(value)
        if self.root is None:
            self.root = _BKNode(value, [payload])
            return
        node = self.root
        while True:
            distance = hamming_distance(value, node.value)
            if distance == 0:
                node.payloads.append(payload)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKNode(value, [payload])
                return
            node = child

    def search(self, value, max_distance):
        if self.root is None:
            return []
        value = int(value)
        results = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = hamming_distance(value, node.value)
            if distance <= max_distance:
                results.extend(
                    (distance, payload) for payload in node.payloads
                )
            lower = distance - max_distance
            upper = distance + max_distance
            stack.extend(
                child
                for edge, child in node.children.items()
                if lower <= edge <= upper
            )
        return results
