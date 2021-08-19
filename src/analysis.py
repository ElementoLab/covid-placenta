#!/usr/bin/env python

"""
Analysis of IHC data from COVID-19 infected placental tissue.
"""

import sys
import typing as tp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tifffile
from tqdm import tqdm
from seaborn_extensions import swarmboxenplot
from stardist.models import StarDist2D
import parmap

from src._config import Config as c
from src.types import Path, DataFrame, Array
from src.ihclib import Image, downcast_int


def main() -> int:
    images, annot = get_files(
        input_dir=c.data_dir, exclude_patterns=["_old", "mask", "C_1", "CD163"]
    )
    annot.to_csv(c.metadata_dir / "image_annotation.csv")

    # Dataset structure
    check_dataset_structure(annot)

    # # Check white balance is the same for all iamges
    # wbos = pd.Series(check_white_balance(images))

    # Segment
    segment(images, annot, overwrite=True)

    # Visualize decompositin and segmentation
    s = annot
    cols = ["marker", "tissue_name", "donor_id"]
    sf = s.groupby(cols).sample(n=1).sort_values(cols).index
    si = [i for i in images if i.fname in sf]
    visualize(si, output_dir=c.results_dir, output_suffix=".representative")

    # Quantify
    _quant = parmap.map(quantify, images, pm_pbar=True)
    quant = pd.concat(_quant).drop_duplicates()
    annot["file"] = [image.fname for image in images]
    quant = (
        quant.reset_index()
        .merge(annot.drop("marker", axis=1), on="file")
        .set_index("cell_id")
    )
    quant.to_csv(c.results_dir / "image_quantification.csv")

    # Read in
    quant = pd.read_csv(c.results_dir / "image_quantification.csv", index_col=0)
    thresh = 0

    # Plot morphology
    fig, axes = plt.subplots(2, 2, figsize=(4.2 * 2, 4 * 2))
    vars_ = [
        "area",
        "major_axis_length",
        "solidity",
        "eccentricity",
    ]
    for ax, var in zip(axes.flat, vars_):
        sns.histplot(quant[var], ax=ax, bins=200, rasterized=True)
    fig.savefig(c.results_dir / "morphology.histplot.svg", **c.figkws)
    plt.close(fig)

    # f = "data/C_1 pt 11267/CD3/FV/CD3 8-20Xf.tiff"
    # f = 'data/C_1 pt 11267/CD3/CP/CD3 14-20Xc.tiff'
    # image = [i for i in images if i.fname == f][0]
    # df = quant.loc[quant["file"] == f]
    # plt.scatter(df["norm_diaminobenzidine"], df["norm_hematoxilyn"])

    # # Filter out outliers?
    # for par in vars_:
    #     a = quant[par]
    #     sel = (a > a.quantile(0.01)) & (a < a.quantile(0.99))
    #     quant = quant.loc[sel]

    # Plot intensity values
    fig, axes = plt.subplots(2, 2, figsize=(4.2 * 2, 4 * 2))
    vars_ = [
        "hematoxilyn",
        "diaminobenzidine",
        "norm_hematoxilyn",
        "norm_diaminobenzidine",
    ]
    colors = ["blue", "orange", "blue", "orange"]
    for ax, var, color in zip(axes.flat, vars_, colors):
        sns.histplot(quant[var], ax=ax, bins=200, rasterized=True, color=color)
    for ax in axes[1, :]:
        ax.axvline(thresh, linestyle="--", color="grey")
    v = max(
        abs(quant["norm_diaminobenzidine"].quantile(0.01)),
        abs(quant["norm_diaminobenzidine"].quantile(0.99)),
    )
    v += v * 0.1
    for ax in axes[1, :]:
        ax.set_xlim((-v, v))
    fig.savefig(c.results_dir / "intensity_values.histplot.svg", **c.figkws)
    plt.close(fig)

    # # Inspect raw intensity per image
    # quant.groupby("image").mean()
    # from imc.graphics import get_grid_dims

    # fig = get_grid_dims(len(set(quant["image"])), return_fig=True)
    # for image, ax in zip(images, fig.axes):
    #     q = quant.query(f"file == '{image.fname}'")
    #     ax.scatter(q["norm_diaminobenzidine"], q["norm_hematoxilyn"], s=5, alpha=0.8)
    #     ax.set(title=image.fname)
    # fig.tight_layout()

    # # Filter out non-cells
    # image = images[0]
    # q = quant.query(f"file == '{image.fname}'")
    # mask1 = image.mask.copy()
    # mask1[np.isin(mask1, q.index[q["norm_hematoxilyn"] < 0])] = 0
    # mask2 = image.mask.copy()
    # mask2[np.isin(mask2, q.index[q["norm_hematoxilyn"] < 0.5])] = 0
    # mask3 = image.mask.copy()
    # mask3[np.isin(mask3, q.index[q["norm_hematoxilyn"] < 1.0])] = 0

    # fig, axes = plt.subplots(3, 2, sharex=True, sharey=True)
    # axes[0][0].imshow(image.image)
    # axes[0][1].imshow(image.mask)
    # h, d = image.decompose_hdab(normalize=False)
    # axes[1][0].imshow(h, cmap="Blues", vmin=0, vmax=0.15)
    # axes[1][1].imshow(d, cmap="Oranges", vmin=0, vmax=0.15)
    # axes[2][0].imshow(mask1)
    # axes[2][1].imshow(mask2)
    # for ax in axes.flat:
    #     ax.axis("off")

    # # Per marker, tissue, donor
    # markers = annot["marker"].unique()
    # tissues = annot["tissue_name"].unique()
    # donors = annot["donor_id"].unique()
    # for var_, label, thresh2, kwargs in [
    #     ("diaminobenzidine", "raw", 0.01, dict()),
    #     ("norm_diaminobenzidine", "norm", 0, dict(xlim=(-3, 3))),
    # ]:
    #     for marker in markers:
    #         print(label, marker)
    #         n, m = len(tissues), len(donors)
    #         fig, axes = plt.subplots(
    #             n, m, figsize=(m * 2, n * 2), sharex=True, sharey=True
    #         )
    #         for i, tissue in enumerate(tissues):
    #             for j, donor in enumerate(donors):
    #                 p = quant.query(
    #                     f"tissue_name == '{tissue}' & donor_id == '{donor}' & marker == '{marker}'"
    #                 )
    #                 if p.empty:
    #                     axes[i, j].axis("off")
    #                     continue
    #                 sns.histplot(
    #                     p[var_],
    #                     stat="probability",
    #                     ax=axes[i, j],
    #                     # bins=50,
    #                     rasterized=True,
    #                 )
    #                 axes[i, j].axvline(thresh2, linestyle="--", color="grey")
    #                 f = (p[var_] > thresh2).sum() / p.shape[0]
    #                 axes[i, j].text(thresh2, 0.25, s=f"{f * 100:.1f}%", ha="left")
    #                 axes[i, j].set(xlabel=None, **kwargs)
    #         for ax, tissue in zip(axes[:, 0], tissues):
    #             ax.set(ylabel=tissue)
    #         for ax, donor in zip(axes[0, :], donors):
    #             ax.set(title=donor)

    #         fig.suptitle(marker)
    #         fig.savefig(
    #             c.results_dir / f"intensity_values.{label}.histplot.{marker}.pdf",
    #             **c.figkws,
    #         )
    #         plt.close(fig)

    # Threshold
    quant["pos"] = quant["norm_diaminobenzidine"] > thresh

    # # Using GMM
    # from imc.ops.mixture import get_population
    # _pos = list()
    # for file in quant['file'].unique():
    #     q = quant.loc[quant['file'] == file].copy()
    #     q["pos"] = get_population(q["diaminobenzidine"])
    #     _pos.append(q)
    #     # print(file, q.shape[0])
    #     # q["pos"] = pd.Categorical(q['pos'])
    #     # fig, stats = swarmboxenplot(data=q, x='pos', y='diaminobenzidine')
    # pos = pd.concat(_pos)

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

    markers = quant["marker"].unique()
    fig, axes = plt.subplots(
        1, len(markers), figsize=(len(markers) * 4, 1 * 4), sharey=True
    )
    # _stats = list()
    for marker, ax in zip(markers, axes.T):
        pp = p.query(f"marker == '{marker}'")
        if pp.empty:
            continue
        swarmboxenplot(
            data=pp,
            x="tissue_name",
            y="percent_positive",
            hue="donor_id",
            test=False,
            swarm=True,
            ax=ax,
            plot_kws=dict(palette="tab20"),
        )
        # _stats.append(stats.assign(marker=marker))
        ax.set_title(marker)
    # stats = pd.concat(_stats)
    # stats.to_csv(c.results_dir / "percent_positive.per_marker.per_patient.statistics.csv")
    fig.savefig(
        c.results_dir / "percent_positive.per_marker.per_patient.swarmboxenplot.svg",
        **c.figkws,
    )

    # Compare COVID/control
    fig, axes = plt.subplots(1, len(markers), figsize=(len(markers) * 4, 1 * 4))
    _stats = list()
    for marker, ax in zip(markers, axes.T):
        pp = p.query(f"marker == '{marker}'")
        if pp.empty:
            continue
        stats = swarmboxenplot(
            data=pp,
            x="tissue_name",
            y="percent_positive",
            hue="disease",
            test=True,
            swarm=True,
            ax=ax,
        )
        _stats.append(stats.assign(marker=marker))
        ax.set_title(marker)
    # for ax in fig.axes:
    #     ax.set_ylim((0, 70))
    stats = pd.concat(_stats)
    stats.to_csv(c.results_dir / "percent_positive.per_marker.per_disease.statistics.csv")
    fig.savefig(
        c.results_dir / "percent_positive.per_marker.per_disease.swarmboxenplot.svg",
        **c.figkws,
    )
    plt.close(fig)

    # # Separately
    p["label"] = p["marker"] + " - " + p["tissue_name"]
    q = p.drop("file", axis=1).pivot_table(
        index="file", columns="label", values="percent_positive"
    )
    fig, stats = swarmboxenplot(
        data=q.join(annot[["disease"]]),
        x="disease",
        y=q.columns,
        test=True,
        swarm=True,
    )
    stats["p-cor"] = pg.multicomp(stats["p-unc"].values, method="fdr_bh")[1]
    stats.to_csv(c.results_dir / "percent_positive.per_marker.per_disease.statistics.csv")
    fig.savefig(
        c.results_dir / "percent_positive.per_marker.per_disease.swarmboxenplot.svg",
        **c.figkws,
    )
    for ax in fig.axes:
        ax.set(ylim=(-5, 105))
    fig.savefig(
        c.results_dir
        / "percent_positive.per_marker.per_disease.swarmboxenplot.fixed_scale.svg",
        **c.figkws,
    )
    plt.close(fig)

    # Visualize decomposition, segmentation and thresholding jointly
    # # Select top/bottom N to visualize
    to_demo = list()
    to_demo += q["CD3 - Fetal Villi"].nlargest(5).index.tolist()
    to_demo += q["CD3 - Fetal Villi"].nsmallest(5).index.tolist()
    to_demo += q["CD3 - Maternal Blood Space"].nlargest(5).index.tolist()
    to_demo += q["CD3 - Maternal Blood Space"].nsmallest(5).index.tolist()
    si = [i for i in images if i.fname in to_demo]
    visualize(
        si,
        c.results_dir,
        quant=quant,
        output_suffix=".representative_differential.top_bottom",
    )

    # Select randomly two images per class to contrast
    to_demo = list()
    for tissue in ["Fetal Villi", "Maternal Blood Space"]:
        for dis in ["Control", "COVID"]:
            sel = (
                annot.query(
                    f"tissue_name == '{tissue}' and disease == '{dis}'",
                    engine="python",
                )
                .sample(n=5)
                .index
            )
            to_demo += sel.tolist()

    si = [i for i in images if i.fname in to_demo]
    visualize(
        si,
        c.results_dir,
        quant=quant,
        output_suffix=".representative_differential.random",
    )

    return 0


def _unpack_data(force: bool = True):
    """
    Function to unpack data from zipfiles and homogeneize file structures across cases.
    Should only be run once if starting from zipfiles downloaded from box.com.
    """
    import shutil
    import zipfile

    for s in ["H_2 pt 14059", "C_4 pt 26799", "C_1 pt 11267"]:
        # Clean slate
        d = c.data_dir / s
        if force:
            shutil.rmtree(d)
        # Extract
        f = d + ".zip"
        if not d.exists() or force:
            with zipfile.ZipFile(f) as zf:
                zf.extractall(d)
        # remove 'extra' files
        for f in d.glob("**/*.czi"):
            f.unlink()
        for f in d.glob("**/.DS_Store"):
            f.unlink()

        # fix duplicated folders
        for marker in d.iterdir():
            for tissue1 in marker.iterdir():
                for tissue2 in tissue1.iterdir():
                    if tissue2.name != tissue1.name:
                        continue
                    for file in tissue2.iterdir():
                        file.replace(tissue1 / file.name)
                    tissue2.rmdir()


def get_files(
    input_dir: Path, exclude_patterns: tp.Sequence[str] = None
) -> tp.Tuple[tp.List[Image], pd.DataFrame]:
    files = sorted(set(input_dir.glob("**/*.tiff")))
    if not files:
        raise FileNotFoundError("Could not find any TIFF files!")
    if exclude_patterns is not None:
        files = [
            f for f in files if not any(pat in f.as_posix() for pat in exclude_patterns)
        ]

    abbrv = {
        "CP": "Chorionic Plate",
        "FV": "Fetal Villi",
        "IVS": "Intervillous space",
        "MB": "Maternal Blood Space",
        "MBS": "Maternal Blood Space",
        # "MBS_IVS": "Maternal Blood Space + Intervillous space",
        "MD": "Maternal Decidua",
    }

    images = list()
    _df = dict()
    for file in files:
        img = Image(file.parent.parent.name, file)
        img.fname = file.as_posix()
        images.append(img)
        _df[file.as_posix()] = file.relative_to(c.data_dir).parts
    df = pd.DataFrame(_df, index=["donor_id", "marker", "tissue", "file"]).T
    assert len(images) == df.shape[0]
    df["magnification"] = df["file"].str.extract(r"(\d\d?[X,x])")[0].str.replace("x", "X")
    assert df["tissue"].isin(abbrv.keys()).all()
    df["tissue_name"] = df["tissue"].replace(abbrv)

    cats = {False: "Control", True: "COVID"}
    df["disease"] = df["donor_id"].str.startswith("H").replace(cats)
    df["disease"] = pd.Categorical(df["disease"], ordered=True, categories=cats.values())

    return (images, df.rename_axis(index="image_file"))


def check_dataset_structure(annot: DataFrame) -> None:
    g = (
        annot.groupby(["donor_id", "marker", "tissue_name"])["file"]
        .nunique()
        .rename("image_number")
    )
    g.to_csv(c.results_dir / "image_summary.per_donor.csv")
    g.mean()  # 18.02777
    g.median()  # 19.0
    g.quantile([0.05, 0.95])  # 6.1, 32.7

    g = annot.groupby(["disease", "marker", "tissue_name"])["file"].nunique()
    g.to_csv(c.results_dir / "image_summary.per_disease.csv")


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


def segment(images, annot, model="2D_versatile_fluo", overwrite: bool = False):
    _model = StarDist2D.from_pretrained(model)
    _model.thresholds = {"prob": 0.5, "nms": 0.3}
    for image in tqdm(images):
        if image.mask_file_name.exists() and not overwrite:
            continue
        # mag = int(annot.loc[image.fname, "magnification"].replace("X", ""))
        if _model.name.endswith("fluo"):
            h, _ = image.decompose_hdab(normalize=False)
            h[h < 0] = 0
            h[h > 0.15] = 0.15
        elif _model.name.endswith("he"):
            h = image.image / 255
        # if mag != 10:
        #     h = resize_magnification(h, mag / 10)
        mask, _ = _model.predict_instances(h)
        # if mag != 10:
        #     mask = resize_magnification(mask, 10 / mag)
        #     mask2 = np.zeros(mask.shape, dtype=np.uint)
        #     for i, n in enumerate(sorted(np.unique(mask))[1:]):
        #         mask2[mask == n] = i
        #     mask = mask2
        mask = downcast_int(mask)
        assert mask.shape == image.image.shape[:-1]
        tifffile.imwrite(image.mask_file_name, mask)


def visualize(
    images: tp.Sequence[Image],
    output_dir: Path,
    output_suffix: str = "",
    quant: pd.DataFrame = None,
) -> None:
    from matplotlib.backends.backend_pdf import PdfPages
    from imc.graphics import get_random_label_cmap

    n = 0 if quant is None else 2
    suffix = ("" if quant is None else ".intensity_positivity") + output_suffix
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
            # "Mean per cell",
            "Normalized intensity per cell",
            "Positivity per cell",
        ]

    with PdfPages(pdf_f) as pdf:
        for image in tqdm(images):
            name = " - ".join(
                [
                    image.image_file_name.parent.parent.parent.name,
                    image.image_file_name.parent.parent.name,
                    image.image_file_name.parent.name,
                    image.image_file_name.stem,
                ]
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
            h, d = image.decompose_hdab(normalize=False)
            axes[1].imshow(h, vmin=0, vmax=0.15, cmap="Blues")
            axes[2].imshow(d, vmin=0, vmax=0.15, cmap="Oranges")
            axes[3].imshow(
                np.ma.masked_array(image.mask, mask=image.mask == 0),
                cmap=get_random_label_cmap(),
            )
            axes[4].imshow(image.image)
            axes[4].contour(image.mask, linewidths=0.2, levels=1)

            if quant is not None:
                # Visualize intensity
                cc = quant.loc[quant["file"] == image.fname].copy()
                mask = image.mask

                # # # raw
                # intens = np.zeros(image.mask.shape, dtype=float)
                # for cell in sorted(np.unique(cc.index))[1:]:
                #     intens[mask == cell] = cc.loc[cell, "diaminobenzidine"]
                # intens = np.ma.masked_array(intens, mask=mask == 0)
                # axes[5].imshow(intens, vmax=0.05, cmap="viridis")

                # # normalized
                intens = np.zeros(image.mask.shape, dtype=float)
                for cell in sorted(np.unique(cc.index))[1:]:
                    intens[mask == cell] = cc.loc[cell, "norm_diaminobenzidine"]
                intens = np.ma.masked_array(intens, mask=intens == 0)
                axes[5].imshow(intens, vmin=-1, vmax=1, cmap="RdBu_r")

                # Visualize positivity
                if "pos" in cc.columns:
                    pos = np.zeros(image.mask.shape, dtype=int)
                    pos[np.isin(image.mask, cc.loc[cc["pos"] == True].index)] = 2
                    pos[np.isin(image.mask, cc.loc[cc["pos"] == False].index)] = 1
                    pos = np.ma.masked_array(pos, mask=pos == 0)
                    axes[6].imshow(pos, cmap="cividis")

                    nneg, npos = cc["pos"].value_counts().reindex([False, True]).fillna(0)
                    axes[6].set_title(
                        f"Positive: {npos}; Negative: {nneg}; %: {(npos / (npos + nneg + 1)) * 100:.2f}",
                        y=-0.1,
                        loc="right",
                        fontsize=4,
                    )

            for ax, lab in zip(axes, labels):
                ax.set(title=lab)
            for ax in axes:
                ax.axis("off")
            plt.figure(fig.number)
            pdf.savefig(**c.figkws)
            plt.close(fig)


def quantify(
    image, normalize_background: bool = True, normalize_illumination: bool = False
) -> DataFrame:
    image.normalize_illumination = normalize_illumination
    f = (c.data_dir / image.image_file_name.relative_to(c.data_dir.absolute())).as_posix()
    q = image.quantify(normalize=False, normalize_background=normalize_background)
    return q.assign(file=f)

    # # Serial:
    # from src.ihclib import quantify_cell_intensity, quantify_cell_morphology
    # _quant = list()
    # image = images[0]
    # for image in tqdm(images[images.index(image) :]):
    #     d = image.decompose_hdab(normalize=True)
    #     q = quantify_cell_intensity(d, image.mask, border_objs=True)
    #     q.columns = ["hematoxilyn", "diaminobenzidine"]
    #     q.index.name = "cell_id"

    #     m0, s0 = d[0].mean(), d[0].std()
    #     m1, s1 = d[1].mean(), d[1].std()
    #     q["norm_hematoxilyn"] = (q["hematoxilyn"] - m0) / s0
    #     q["norm_diaminobenzidine"] = (q["diaminobenzidine"] - m1) / s1

    #     m = quantify_cell_morphology(image.mask, border_objs=True)
    #     q = q.join(m)
    #     _quant.append(q.assign(file=image.fname))


if __name__ == "__main__" and "get_ipython" not in locals():
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit()
