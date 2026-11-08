"""ExploreToM (Meta, CC BY-NC 4.0): 200 questions. Waiting on the data file.

The release file has to be downloaded by hand (see shared/benchmarks/data/exploretom/README.md)
and is checked against its published sha256. The question types are chosen after inspecting it.
"""

from shared.benchmarks.common import FROZEN


SOURCE = FROZEN / "exploretom" / "ExploreToM-data-sample.csv"
SOURCE_SHA256 = "f8462b1a2199819a2baef5ab321b7b9240249fe229b94c86d8d06c2c47fc448b"
SOURCE_URL = ("https://huggingface.co/datasets/facebook/ExploreToM/resolve/"
              "a4ac6f257e0034945f829716047ae6306dc625a0/ExploreToM-data-sample.csv")


def load_all(source=SOURCE):
    raise NotImplementedError("ExploreToM loader is written once the data file is inspected")


def freeze_selection():
    raise NotImplementedError("ExploreToM question types are not chosen yet")
