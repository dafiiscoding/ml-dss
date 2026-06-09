import io
import unittest

from PIL import Image, ImageDraw

from src.image_hashing import (
    HammingBKTree,
    hamming_distance,
    perceptual_hash,
)


def _pattern_image():
    image = Image.new("RGB", (96, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 44, 48), fill="navy")
    draw.ellipse((50, 12, 86, 52), fill="orange")
    draw.line((0, 63, 95, 0), fill="black", width=4)
    return image


class PerceptualHashTests(unittest.TestCase):
    def test_hash_is_stable_under_resize_and_jpeg_compression(self):
        original = _pattern_image()
        buffer = io.BytesIO()
        original.resize((192, 128)).save(
            buffer, format="JPEG", quality=70
        )
        buffer.seek(0)
        compressed = Image.open(buffer)
        distance = hamming_distance(
            perceptual_hash(original), perceptual_hash(compressed)
        )
        self.assertLessEqual(distance, 4)

    def test_bk_tree_finds_hashes_within_radius(self):
        tree = HammingBKTree()
        tree.add(0b0000, "exact")
        tree.add(0b0011, "near")
        tree.add(0b1111, "far")
        found = sorted(tree.search(0b0001, max_distance=1))
        self.assertEqual(found, [(1, "exact"), (1, "near")])


if __name__ == "__main__":
    unittest.main()
