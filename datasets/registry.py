"""Catalog of handwriting datasets this project can use, and how each one is
acquired. "auto" sources are fetched by scripts/prepare_dataset.py with no
credentials. "manual" sources require the user to register with the dataset
provider and download the archive themselves (see `instructions`); this
project cannot and does not automate account creation or license agreements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Acquisition = Literal["auto", "manual"]


@dataclass(frozen=True)
class DatasetSourceInfo:
    name: str
    description: str
    acquisition: Acquisition
    homepage_url: str
    instructions: str


REGISTRY: dict[str, DatasetSourceInfo] = {
    "synthetic": DatasetSourceInfo(
        name="synthetic",
        description="Procedurally rendered character/word images from system fonts, "
        "with augmentation. No external download; always available.",
        acquisition="auto",
        homepage_url="",
        instructions="Generated locally by datasets/sources/synthetic.py — nothing to download.",
    ),
    "mnist": DatasetSourceInfo(
        name="mnist",
        description="70k handwritten digit images (0-9), 28x28 grayscale.",
        acquisition="auto",
        homepage_url="https://ossci-datasets.s3.amazonaws.com/mnist/",
        instructions="Downloaded automatically by scripts/prepare_dataset.py mnist.",
    ),
    "emnist": DatasetSourceInfo(
        name="emnist",
        description="Extended MNIST: digits + uppercase + lowercase letters, "
        "28x28 grayscale. The 'byclass' split is used (62 classes).",
        acquisition="auto",
        homepage_url="https://www.nist.gov/itl/products-and-services/emnist-dataset",
        instructions="Downloaded automatically by scripts/prepare_dataset.py emnist "
        "(~550MB archive — only run this when you actually want the data).",
    ),
    "iam": DatasetSourceInfo(
        name="iam",
        description="IAM Handwriting Database: full sentences/lines/words from ~650 "
        "writers, with transcripts.",
        acquisition="manual",
        homepage_url="https://fki.tic.heia-fr.ch/databases/iam-handwriting-database",
        instructions=(
            "Register for a free account at the FKI site above, download "
            "'lines.tgz' (or 'words.tgz') and 'ascii.tgz', then extract both under "
            "datasets/raw/iam/ (so datasets/raw/iam/lines/ and datasets/raw/iam/ascii/ "
            "exist). Run scripts/prepare_dataset.py iam afterward."
        ),
    ),
    "cvl": DatasetSourceInfo(
        name="cvl",
        description="CVL Database: handwritten text from 311 writers, English and German.",
        acquisition="manual",
        homepage_url="https://cvl.tuwien.ac.at/research/cvl-databases/an-off-line-database-for-writer-retrieval-writer-identification-and-word-spotting/",
        instructions=(
            "Download the CVL database archive from the University of Vienna site above "
            "(free, but requires accepting their terms) and extract it under "
            "datasets/raw/cvl/. Run scripts/prepare_dataset.py cvl afterward."
        ),
    ),
    "nist_sd19": DatasetSourceInfo(
        name="nist_sd19",
        description="NIST Special Database 19: handwritten digits and letters, "
        "the source EMNIST was itself derived from.",
        acquisition="manual",
        homepage_url="https://www.nist.gov/srd/nist-special-database-19",
        instructions="Request access via the NIST page above; place the extracted "
        "archive under datasets/raw/nist_sd19/. Not yet implemented — see docs/ROADMAP.md.",
    ),
    "rimes": DatasetSourceInfo(
        name="rimes",
        description="RIMES: French handwritten mail/letters dataset.",
        acquisition="manual",
        homepage_url="https://teklia.com/research/rimes-database/",
        instructions="Requires a license agreement with the RIMES consortium; place "
        "the extracted archive under datasets/raw/rimes/. Not yet implemented.",
    ),
    "khatt": DatasetSourceInfo(
        name="khatt",
        description="KHATT: Arabic handwritten text database.",
        acquisition="manual",
        homepage_url="https://khatt.ideas2serve.net/",
        instructions="Requires registration; place the extracted archive under "
        "datasets/raw/khatt/. Not yet implemented — out of scope unless Arabic-script "
        "support is prioritized.",
    ),
    "bentham": DatasetSourceInfo(
        name="bentham",
        description="Bentham Papers: 18th/19th century English manuscript pages.",
        acquisition="manual",
        homepage_url="https://zenodo.org/records/44519",
        instructions="Publicly downloadable from Zenodo; place the extracted archive "
        "under datasets/raw/bentham/. Not yet implemented.",
    ),
    "casia": DatasetSourceInfo(
        name="casia",
        description="CASIA-HWDB: Chinese handwritten character/text database.",
        acquisition="manual",
        homepage_url="http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html",
        instructions="Requires registration with CASIA; place the extracted archive "
        "under datasets/raw/casia/. Not yet implemented — out of scope unless "
        "Chinese-script support is prioritized.",
    ),
}


def get_source(name: str) -> DatasetSourceInfo:
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[name]
