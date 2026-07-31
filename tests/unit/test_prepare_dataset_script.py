import pytest

from datasets.sources.cvl import CvlDatasetSource
from datasets.sources.emnist import EmnistDatasetSource
from datasets.sources.iam import IamDatasetSource
from datasets.sources.mnist import MnistDatasetSource
from datasets.sources.synthetic import SyntheticDatasetSource
from scripts.prepare_dataset import build_source


@pytest.mark.parametrize(
    "name,expected_type",
    [
        ("synthetic", SyntheticDatasetSource),
        ("mnist", MnistDatasetSource),
        ("emnist", EmnistDatasetSource),
        ("iam", IamDatasetSource),
        ("cvl", CvlDatasetSource),
    ],
)
def test_build_source_returns_expected_type(name, expected_type):
    source = build_source(name, max_samples=None)
    assert isinstance(source, expected_type)


def test_build_source_raises_for_unimplemented_dataset():
    with pytest.raises(NotImplementedError):
        build_source("nist_sd19", max_samples=None)


def test_build_source_passes_max_samples_to_mnist():
    source = build_source("mnist", max_samples=42)
    assert source._max_samples == 42
