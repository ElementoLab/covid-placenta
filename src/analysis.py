#!/usr/bin/env python

"""
Analysis description.
"""

import sys
import typing as tp

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from csbdeep.utils import normalize
import tifffile
from tqdm import tqdm
from seaborn_extensions import clustermap, swarmboxenplot
from stardist.models import StarDist2D

from src._config import Config as c
from src.types import Path
from src.ihclib import Image, Analysis

Array = tp.Union[np.ndarray]


def main() -> int:
    images, annot = get_files(c.data_dir)
    annot.to_csv(c.metadata_dir / "image_annotation.csv")

    # Check white balance is the same for all iamges
    wbos = pd.Series(check_white_balance(images))
    # assert wbos.nunique() == 1

    # Segment
    # model = StarDist2D.from_pretrained("2D_versatile_he")
    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    model.thresholds = {"prob": 0.5, "nms": 0.3}
    for image in tqdm(images):
        image.normalize_illumination = False
        if image.mask_file_name.exists():
            continue
        mag = int(annot.loc[image.fname, "magnification"].replace("X", ""))
        # if mag == 10:
        #     continue
        if model.name.endswith("fluo"):
            h = image.decompose_hdab()[0]
            h[h < 0] = 0
            h = h / 0.15
        elif model.name.endswith("he"):
            h = image.image / 255
        if mag != 10:
            h = resize_magnification(h, mag / 10)
        mask, _ = model.predict_instances(h)
        if mag != 10:
            mask = resize_magnification(mask, 10 / mag)
            mask2 = np.zeros(mask.shape, dtype=np.uint)
            for i, n in enumerate(sorted(np.unique(mask))[1:]):
                mask2[mask == n] = i
            mask = mask2
        assert mask.shape == image.image.shape[:-1]
        tifffile.imwrite(image.mask_file_name, mask)

    # Visualize decompositin and segmentation
    visualize(images, c.results_dir)

    for image in images:
        q = (
            c.data_dir / image.image_file_name.relative_to(c.data_dir.absolute())
        ).as_posix()
        if q == n:
            break

    # Quantify
    _quant = list()
    image = images[0]
    for image in tqdm(images[images.index(image) :]):
        f = (
            c.data_dir / image.image_file_name.relative_to(c.data_dir.absolute())
        ).as_posix()
        q = image.quantify()[["hematoxilyn", "diaminobenzidine"]]
        d = image.decompose_hdab(normalize=False)
        m0, s0 = d[0].mean(), d[0].std()
        m1, s1 = d[1].mean(), d[1].std()
        q["norm_hematoxilyn"] = (q["hematoxilyn"] - m0) / s0
        q["norm_diaminobenzidine"] = (q["diaminobenzidine"] - m1) / s1
        _quant.append(q.assign(file=f))
    quant = pd.concat(_quant).drop_duplicates()
    annot["file"] = [
        (c.data_dir / image.image_file_name.relative_to(c.data_dir.absolute())).as_posix()
        for image in images
    ]
    quant = quant.reset_index().merge(annot, on="file")
    quant.to_csv(c.results_dir / "image_quantification.csv", index=False)

    quant = pd.read_csv(c.results_dir / "image_quantification.csv", index_col=0)

    # Plot intensity values
    fig, axes = plt.subplots(2, 2, figsize=(4.2 * 2, 4 * 2))
    markers = [
        "hematoxilyn",
        "diaminobenzidine",
        "norm_hematoxilyn",
        "norm_diaminobenzidine",
    ]
    colors = ["blue", "orange", "blue", "orange"]
    for ax, var, color in zip(axes.flat, markers, colors):
        sns.histplot(quant[var], ax=ax, bins=200, rasterized=True, color=color)
    fig.savefig(c.results_dir / "intensity_values.histplot.svg", **c.figkws)

    # Threshold
    quant["pos"] = quant["norm_diaminobenzidine"] > 0

    # Percent positive per image
    total_count = quant.groupby(["file"])["pos"].count().to_frame("cells")
    pos_count = quant.groupby(["file"])["pos"].sum().to_frame("positive")

    p = total_count.join(pos_count).join(annot)
    p["percent_positive"] = (p["positive"] / p["cells"]) * 100
    p.to_csv(c.results_dir / "percent_positive.per_marker.csv")
    locations = [
        "Chorionic Plate",
        "Fetal Villi",
        "Maternal Blood Space",
        "Maternal Decidua",
    ]
    p = p.loc[p["tissue_name"].isin(locations)]

    markers = annot["marker"].unique()
    fig, axes = plt.subplots(
        1, len(markers), figsize=(len(markers) * 4, 1 * 4), sharey=True
    )
    # _stats = list()
    for marker, ax in zip(markers, axes.T):
        pp = p.query(f"marker == '{marker}'")
        # stats = swarmboxenplot(
        swarmboxenplot(
            data=pp,
            x="tissue_name",
            y="percent_positive",
            hue="patient_id",
            test=False,
            swarm=True,
            ax=ax,
        )
        # _stats.append(stats.assign(marker=marker))
        ax.set_title(marker)
    # stats = pd.concat(_stats)
    # stats.to_csv(c.results_dir / "percent_positive.per_marker.statistics.csv")
    fig.savefig(
        c.results_dir / "percent_positive.per_marker.swarmboxenplot.svg", **c.figkws
    )

    # Visualize decomposition, segmentation and thresholding jointly
    visualize(images, c.results_dir, quant)

    return 0


def _fix_file_structure():
    """
    Function to homogeneize file structures across cases.
    Should only be run once.
    """
    import os
    import shutil

    # Clean slate
    d = c.data_dir / r"H_1\ 7730"
    os.system(f"rm -r {d}")
    # Extract
    f = c.data_dir / r"H_1\ 7730.zip"
    os.system(f"unzip -d {d} {f}")
    # remove 'extra' files
    os.system(f"find {d} -name '.DS_Store' -delete")
    os.system(f"find {d} -name '*.czi' -delete")
    # remove unpaired files
    os.system(f"find {d} -name 'CD 10-20Xd.tiff_metadata.xml' -delete")
    os.system(f"find {d} -name 'CD3 7-10Xd.tiff_metadata.xml' -delete")
    for marker in (c.data_dir / "H_1 7730").iterdir():
        for tissue in sorted([f for f in marker.iterdir() if f.is_dir()]):
            for file in tissue.iterdir():
                end = ".tiff" + file.as_posix().split(".tiff")[1]
                s = file.stem.replace(" d side", "_d_side").replace(" c  side", "_c_side")
                p = s.split(" ")[1:]
                n, mag = p if "-" not in p[0] else p[0].split("-")
                mag = mag.split(".")[0]
                new = tissue + f" {n} {marker.name}-{mag}{end}"
                assert not new.exists()
                file.replace(new)
            shutil.rmtree(tissue)


def get_files(
    input_dir: Path, exclude_patterns: tp.Sequence[str] = None
) -> tp.Tuple[tp.List[Image], pd.DataFrame]:
    # from aicsimageio import AICSImage
    # from aicsimageio.readers.czi_reader import CziReader
    files = sorted(list(input_dir.glob("**/*.tiff")))
    if exclude_patterns is not None:
        files = [
            f for f in files for pat in exclude_patterns if not any([pat in f.as_posix()])
        ]

    abbrv = {
        "CP": "Chorionic Plate",
        "FV": "Fetal Villi",
        "IVS": "Intervillous space",
        "MBS": "Maternal Blood Space",
        "MBS_IVS": "Maternal Blood Space + Intervillous space",
        "MD": "Maternal Decidua",
    }

    images = list()
    _df = dict()
    for file in files:
        img = Image(file.parent.name, file)
        img.fname = file.as_posix()
        images.append(img)
        parts = (
            file.stem.strip()
            .replace("  ", " ")
            .replace("MBS + IVS", "MBS_IVS")
            .replace("MBS + IV", "MBS_IVS")
            .split(" ")
        )
        if len(parts) == 2:
            parts = (
                file.stem.strip()
                .replace("  ", " ")
                .replace("MBS + IVS", "MBS_IVS")
                .replace("MBS + IV", "MBS_IVS")
                .replace("-", " -")
                .split(" ")
            )
        parts = parts[:3]
        _df[file.as_posix()] = parts + [file.parent.name, file.parent.parent.name]
    df = pd.DataFrame(
        _df, index=["tissue", "id", "magnification", "marker", "patient_id"]
    ).T.rename_axis(index="image_file")
    df["id"] = df["id"].str.replace(r"-.*", "", regex=True)
    df["magnification"] = (
        df["magnification"]
        .str.replace(r".*-", "", regex=True)
        .str.replace(r"X.*", "", regex=True)
        .replace("m_d_side", "20")
        .replace("m_c_side", "20")
        + "X"
    )
    df["tissue_name"] = df["tissue"].replace(abbrv)
    return images, df


def check_white_balance(images: tp.Sequence[Image]) -> tp.Dict[str, float]:
    import re

    # from bs4 import BeautifulSoup

    _wbos = dict()
    for image in images:
        xml_f = image.image_file_name + "_metadata.xml"
        # soup = BeautifulSoup(xml_f.open().read(), 'xml')
        # soup.find_all('WhiteBalanceOffset')
        try:
            content = xml_f.open().read()
        except FileNotFoundError:
            print(f"Image '{image.image_file_name}' is missing XML annotation.")
            continue
        wbo_s = [l for l in content.split("\n") if "WhiteBalanceOffset " in l][0].strip()
        _wbos[image.name] = float(re.findall(r"\>(.*?)\</", wbo_s)[0])
    return _wbos


def resize_magnification(arr: Array, fraction: float) -> Array:
    import skimage

    assert fraction > 0

    if len(arr.shape) == 2:
        arr = arr[..., np.newaxis]

    if fraction < 1:
        f = int(1 / fraction)
    else:
        f = int(abs(fraction))
    x, y = arr.shape[0] // f, arr.shape[1] // f
    if fraction < 1:
        out = arr[:x, :y, ...]
        out_shape = arr.shape
        out = skimage.transform.resize(out, out_shape, order=0)
    else:
        out = np.zeros(arr.shape)
        out_shape = (x, y, arr.shape[2])
        out[:x, :y, ...] = skimage.transform.resize(
            arr, out_shape, anti_aliasing=True, order=0
        )
    return out.squeeze()


def visualize(
    images: tp.Sequence[Image], output_dir: Path, quant: pd.DataFrame = None
) -> None:
    from matplotlib.backends.backend_pdf import PdfPages
    from imc.graphics import get_random_label_cmap

    n = 0 if quant is None else 3
    suffix = "" if quant is None else ".intensity_positivity"
    pdf_f = output_dir.mkdir() / f"segmentation{suffix}.pdf"
    labels = [
        "IHC",
        # "IHC - illumination balanced",
        "Hematoxilyn",
        "Diaminobenzidine",
        "Nuclei segmentation",
        "Overlay",
    ]
    if quant is not None:
        labels += [
            "Mean per cell",
            "Normalized intensity per cell",
            "Positivity per cell",
        ]

    with PdfPages(pdf_f) as pdf:
        for image in tqdm(images):
            name = (
                image.image_file_name.parent.parent.name
                + " - "
                + image.image_file_name.stem
            )
            fig, axes = plt.subplots(
                1,
                5 + n,
                figsize=(4 * 5, 4),
                sharex=True,
                sharey=True,
                gridspec_kw=dict(wspace=0),
            )
            axes[0].set_title(name, y=0, pad=-0.25, loc="left", rotation=90, fontsize=6)
            # axes[0].imshow(tifffile.imread(image.image_file_name))
            # axes[1].imshow(image.image)
            axes[0].imshow(image.image)
            d = image.decompose_hdab(normalize=False)
            axes[1].imshow(d[0], vmin=0, vmax=0.15, cmap="Blues")
            axes[2].imshow(d[1], vmin=0, vmax=0.15, cmap="Oranges")
            axes[3].imshow(
                np.ma.masked_array(image.mask, mask=image.mask == 0),
                cmap=get_random_label_cmap(),
            )
            axes[4].imshow(image.image)
            axes[4].contour(image.mask, linewidths=0.2, levels=1)

            if quant is not None:
                # Visualize intensity
                f = c.data_dir / image.image_file_name.relative_to(c.data_dir.absolute())
                cc = quant.loc[quant["file"] == f.as_posix()].copy()
                mask = image.mask
                intens = np.zeros(image.mask.shape, dtype=float)
                for cell in sorted(np.unique(cc.index))[1:]:
                    intens[mask == cell] = cc.loc[cell, "diaminobenzidine"]
                axes[5].imshow(intens, vmax=cc["diaminobenzidine"].max(), cmap="viridis")

                # Normalize intensity
                m, s = d[1].mean(), d[1].std()
                intens = np.zeros(image.mask.shape, dtype=float)
                for cell in sorted(np.unique(cc.index))[1:]:
                    intens[mask == cell] = (cc.loc[cell, "diaminobenzidine"] - m) / s
                intens = np.ma.masked_array(intens, mask=intens == 0)
                axes[6].imshow(intens, vmin=-1, vmax=1, cmap="RdBu_r")

                # Visualize positivity
                axes[7].imshow(intens > 0, cmap="cividis")
                npos = len(np.unique(mask[intens >= 0.25]))
                nneg = len(np.unique(mask[intens < 0.25]) - 1)
                axes[7].set_title(
                    f"Positive: {npos}; Negative: {nneg}; %: {(npos / (npos + nneg)) * 100:.2f}",
                    y=-0.1,
                    loc="right",
                    fontsize=4,
                )
                # cc["pos"] = cc["diaminobenzidine"] > 0.0225
                # pos = cc.query("pos == True").index.astype(int).tolist()
                # neg = cc.query("pos != True").index.astype(int).tolist()
                # # image = [i for i in images if i.image_file_name == pf][0]
                # pmask = np.zeros(mask.shape, dtype=np.uint)
                # pmask[np.isin(mask, pos)] = 2
                # pmask[np.isin(mask, neg)] = 1
                # axes[7].imshow(pmask, vmax=2, cmap="cividis")

            for ax, lab in zip(axes, labels):
                ax.set(title=lab)
            for ax in axes:
                ax.axis("off")
            plt.figure(fig)
            pdf.savefig(**c.figkws)
            plt.close(fig)


if __name__ == "__main__" and "get_ipython" not in locals():
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit()
