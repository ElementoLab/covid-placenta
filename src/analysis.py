#!/usr/bin/env python

"""
Analysis of IHC data from COVID-19 infected placental tissue.
"""

import sys
import typing as tp

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import tifffile
from tqdm import tqdm
from seaborn_extensions import swarmboxenplot
import pingouin as pg
from stardist.models import StarDist2D
import parmap

from src._config import Config as c
from src.types import Path, DataFrame, Array
from src.ihclib import Image, downcast_int


def main() -> int:
    images, annot = get_files(
        input_dir=c.data_dir, exclude_patterns=["_old", "mask", "fluo"]
    )
    # C_1 exclude all but CD163
    annot = annot.query(
        """(donor_id != 'C_1 pt 11267') or ((donor_id == 'C_1 pt 11267') and (marker == 'CD163'))"""
    )
    images = [i for i in images if i.fname in annot.index]
    annot.to_csv(c.metadata_dir / "image_annotation.csv")

    # Dataset structure
    check_dataset_structure(annot)

    # Segment
    segment(images, annot, overwrite=False)

    # Visualize decompositin and segmentation # # one random image per group
    cols = ["marker", "tissue_name", "donor_id"]
    sf = annot.groupby(cols).sample(n=1).sort_values(cols).index
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
    quant = filter_cells_based_on_annotations(quant)
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

    # Threshold
    quant["pos"] = quant["norm_diaminobenzidine"] > thresh

    # Percent positive per image
    total_count = quant.groupby(["file"])["pos"].count().to_frame("cells")
    pos_count = quant.groupby(["file"])["pos"].sum().to_frame("positive")

    p = total_count.join(pos_count).join(annot.drop("file", axis=1))
    p["percent_positive"] = (p["positive"] / p["cells"]) * 100
    p.to_csv(c.results_dir / "percent_positive.per_marker.csv")

    # Read in
    p = pd.read_csv(c.results_dir / "percent_positive.per_marker.csv", index_col=0)
    p["disease"] = pd.Categorical(
        p["disease"], ordered=True, categories=["Control", "COVID"]
    )
    p["tissue_name"] = pd.Categorical(
        p["tissue_name"], ordered=True, categories=c.tissues
    )
    p = p.loc[p["cells"] >= 10]

    # Plot
    # # Jointly, per marker
    markers = p["marker"].unique()
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
        ax.set_title(marker)
    fig.savefig(
        c.results_dir / "percent_positive.per_marker.per_patient.swarmboxenplot.svg",
        **c.figkws,
    )

    # # Jointly, per tissue
    fig, axes = plt.subplots(1, len(c.tissues), figsize=(len(c.tissues) * 4, 1 * 3.7))
    for tissue, ax in zip(c.tissues, axes.T):
        pp = p.query(f"tissue_name == '{tissue}'")
        if pp.empty:
            continue
        swarmboxenplot(
            data=pp,
            x="disease",
            y="percent_positive",
            hue="marker",
            test=False,
            swarm=True,
            ax=ax,
            plot_kws=dict(palette="Set1"),
        )
        ax.set_title(tissue)
    fig.savefig(
        c.results_dir / "percent_positive.per_tissue.per_marker.swarmboxenplot.svg",
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
    p["label"] = p["marker"].astype(str) + " - " + p["tissue_name"].astype(str)
    cats = [m + " - " + t for m in markers for t in c.tissues]
    # p["label"] = pd.Categorical(p["label"], ordered=True, categories=cats)
    q = p.pivot_table(index="file", columns="label", values="percent_positive")[cats]

    fig, stats = swarmboxenplot(
        data=q.join(annot[["disease"]]),
        x="disease",
        y=q.columns,
        test=True,
        swarm=True,
        plot_kws=dict(size=3, alpha=0.2),
        fig_kws=dict(nrows=4, ncols=4, figsize=(7, 12), sharey="col"),
    )
    # for ax in fig.axes:
    #     ax.set_ylim(bottom=-5)
    stats["p-cor"] = pg.multicomp(stats["p-unc"].values, method="fdr_bh")[1]
    stats.to_csv(c.results_dir / "percent_positive.per_marker.per_disease.statistics.csv")
    fig.savefig(
        c.results_dir
        / "percent_positive.per_marker.per_disease.swarmboxenplot.separate.fixed_scale_per_tissue.svg",
        **c.figkws,
    )
    for ax in fig.axes:
        ax.set(ylim=(-5, 105))
    fig.savefig(
        c.results_dir
        / "percent_positive.per_marker.per_disease.swarmboxenplot.fixed_scale.svg",
        **c.figkws,
    )
    plt.close("all")

    for marker in markers:
        fig, stats = swarmboxenplot(
            data=p.query(f"marker == '{marker}'"),
            x="tissue_name",
            y="percent_positive",
            hue="disease",
            test=True,
            swarm=True,
            plot_kws=dict(size=3, alpha=0.2),
        )
        fig.axes[0].set(title=marker)
        fig.savefig(
            c.results_dir / f"percent_positive.{marker}.per_disease.swarmboxenplot.svg",
            **c.figkws,
        )
        plt.close(fig)

    # As heatmap
    # # Absolute values
    pp = p.pivot_table(
        index="marker", columns=["disease", "tissue_name"], values="percent_positive"
    )
    kws = dict(square=True, cbar_kws=dict(label="% positive cells"))
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.heatmap(pp, **kws, ax=ax)
    fig.savefig(
        c.results_dir
        / "percent_positive.per_marker.per_tissue.absolute.heatmap.by_disease.svg",
        **c.figkws,
    )
    plt.close(fig)

    ppz = ((pp.T - pp.mean(1)) / pp.std(1)).T
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.heatmap(ppz, **kws, ax=ax, cmap="RdBu_r", center=0)
    fig.savefig(
        c.results_dir
        / "percent_positive.per_marker.per_tissue.absolute.heatmap.by_disease.zscore.svg",
        **c.figkws,
    )
    plt.close(fig)

    pp = p.pivot_table(
        index="marker", columns=["tissue_name", "disease"], values="percent_positive"
    )

    fig, ax = plt.subplots(figsize=(3, 3))
    sns.heatmap(pp, **kws, ax=ax)
    fig.savefig(
        c.results_dir
        / "percent_positive.per_marker.per_tissue.absolute.heatmap.interleaved.svg",
        **c.figkws,
    )
    plt.close(fig)

    ppz = ((pp.T - pp.mean(1)) / pp.std(1)).T
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.heatmap(ppz, **kws, ax=ax, cmap="RdBu_r", center=0)
    fig.savefig(
        c.results_dir
        / "percent_positive.per_marker.per_tissue.absolute.heatmap.interleaved.zscore.svg",
        **c.figkws,
    )
    plt.close(fig)

    # # Fold-changes
    stats = stats.join(
        stats["Variable"]
        .str.split(" - ")
        .apply(pd.Series)
        .rename(columns={0: "marker", 1: "tissue"})
    )
    fc = stats.pivot_table(index="marker", columns="tissue", values="hedges")
    fc = fc[c.tissues] * -1
    pvals = stats.pivot_table(index="marker", columns="tissue", values="p-cor")
    pvals = ((pvals < 0.05) & (pvals > 0.01)).replace({True: 1}) + (
        (pvals < 0.01)
    ).replace({True: 2})

    fig, ax = plt.subplots(figsize=(3, 3))
    sns.heatmap(
        fc,
        ax=ax,
        center=0,
        cmap="RdBu_r",
        square=True,
        cbar_kws=dict(label="Log fold change (COVID / Control)"),
        annot=pvals,
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    r = {"0": "", "1": "*", "2": "**"}
    for e in ax.get_children():
        if isinstance(e, matplotlib.text.Text):
            if e.get_text() in r:
                e.set_text(r[e.get_text()])
    fig.savefig(
        c.results_dir / "percent_positive.per_marker.per_tissue.fold_change.heatmap.svg",
        **c.figkws,
    )
    plt.close(fig)

    return 0


def filter_cells_based_on_annotations(quant, plot=False):
    """
    Images with CD163 stain on maternal blood space may contain parts
    of fetal vili, so polygons were drawn manually to identify the blood space only.
    """
    import json

    from matplotlib.backends.backend_pdf import PdfPages

    if plot:
        pdf = PdfPages(c.results_dir / "annotation_visualizations.pdf")

    quant = quant.reset_index()

    shape = tifffile.imread(quant.iloc[0]["file"]).shape[:2]
    sel = (quant["marker"] == "CD163") & (quant["tissue_name"] == "Maternal Blood Space")
    files = quant.loc[sel, "file"].drop_duplicates().sort_values()

    _quant = list()
    jsons = dict()
    for file in tqdm(files):
        jf = Path(file.replace(".tiff", ".json"))
        if not jf.exists():
            continue

        j = json.load(jf.open())
        jsons[file] = j["shapes"]
        tmasks = dict(B=np.zeros(shape, dtype=int), V=np.zeros(shape, dtype=int))
        for poly in j["shapes"]:
            label = poly["label"]
            if label not in ["V", "B"]:
                continue
            tmasks[label] += polygon_to_mask(poly["points"], shape[::-1]).astype(int)
        tmask = (tmasks["B"] > 0).astype(int) - (tmasks["V"] > 0).astype(int)

        # Overlay cell mask with topological mask
        mask_f = file.replace(".tiff", ".stardist_mask.tiff")
        mask = tifffile.imread(mask_f)
        sel_cells = np.unique(mask[(mask > 0) & (tmask > 0)])
        _quant.append(
            quant.loc[sel & (quant["file"] == file) & (quant["cell_id"].isin(sel_cells))]
        )

        if plot:
            fig, axes = plt.subplots(1, 4, figsize=(4 * 4, 4))
            axes[0].imshow(tifffile.imread(file))
            axes[1].imshow(mask)
            axes[2].imshow(tmask)
            mask[~np.isin(mask, sel_cells)] = 0
            axes[3].imshow(mask)
            axes[0].text(0, shape[0] + 100, s="/".join(file.split("/")[1:]))
            labels = [
                "Original image",
                "Cellular mask",
                "Tissue mask",
                "Cellular mask\nin Maternal Blood Space",
            ]
            for ax, label in zip(axes, labels):
                ax.set(title=label)
                ax.axis("off")
            pdf.savefig(fig, **c.figkws)
            plt.close(fig)
    if plot:
        pdf.close()

    json.dump(
        jsons, open(c.metadata_dir / "image_polygon_annotations.json", "w"), indent=4
    )

    return quant.loc[~sel].append(pd.concat(_quant)).sort_values(["file"])


def plot_deconvolution():
    from seaborn_extensions import clustermap

    opts = [
        ("CYBERSORT", "placenta.deconv.v2.xlsx", 100, dict(nrows=16)),
    ]
    for name, fname, factor, kws in opts:
        oprefix = c.results_dir / f"cellular_deconvolution.{name}."
        dec = pd.read_excel(c.metadata_dir / "original" / fname, index_col=0, **kws)
        dec *= factor
        annot = dec.columns.to_series().str.split("_").apply(pd.Series).drop(0, axis=1)
        annot.columns = ["sample_group", "sample_id"]
        annot["sample_group"] = pd.Categorical(
            annot["sample_group"],
            ordered=True,
            categories=["Control", "Inflammatory", "Positive", "Borderline", "High"],
        )

        if name == "CYBERSORT":
            mean = dec.T.groupby(
                annot["sample_group"].replace("Borderline", "Positive")
            ).mean()
            v = mean.values.max()
            v += v * 0.1
            for cmap in [
                "summer",
                "hot",
                "afmhot",
                "copper",
                "viridis",
                "plasma",
                "magma",
                "cividis",
                "Reds",
                "OrRd",
                "Oranges",
                "RdPu",
                "BuPu",
                "BuGn",
            ]:
                grid = clustermap(
                    mean.sort_index(ascending=False),
                    cbar_kws=dict(label="Inferred cellular\ncomposition"),
                    dendrogram_ratio=0.1,
                    row_cluster=False,
                    vmax=v,
                    figsize=(5, 4),
                    cmap=cmap,
                )
                grid.fig.savefig(
                    oprefix + f"clustermap.no_borderline.{cmap}.pdf", **c.figkws
                )
        else:
            mean = dec.T.groupby(annot["sample_group"]).mean()
            grid = clustermap(
                mean,
                cbar_kws=dict(label="Inferred cellular\ncomposition"),
                dendrogram_ratio=0.1,
                cmap="RdBu_r",
                center=0,
                figsize=(5, 4.5),
            )
            grid.fig.savefig(oprefix + "clustermap.svg", **c.figkws)

            mean = dec.T.groupby(
                annot["sample_group"].replace("Borderline", "Positive")
            ).mean()
            grid = clustermap(
                mean,
                cbar_kws=dict(label="Inferred cellular\ncomposition"),
                dendrogram_ratio=0.1,
                cmap="RdBu_r",
                center=0,
                figsize=(5, 4),
            )
            grid.fig.savefig(oprefix + "clustermap.no_borderline.svg", **c.figkws)

        p = dec.T.join(annot)
        fig, stats = swarmboxenplot(
            data=p,
            x="sample_group",
            y=dec.index,
            plot_kws=dict(paletter="inferno"),
            fig_kws=dict(figsize=(7, 7)),
        )
        fig.savefig(oprefix + "swarmboxenplot.svg", **c.figkws)

        q = annot.copy()
        q.loc[q["sample_group"] == "Borderline", "sample_group"] = "Positive"
        q["sample_group"] = q["sample_group"].cat.remove_unused_categories()
        p = dec.T.join(q)
        fig, stats = swarmboxenplot(
            data=p,
            x="sample_group",
            y=dec.index,
            plot_kws=dict(paletter="inferno"),
            fig_kws=dict(figsize=(7, 7)),
        )
        fig.savefig(
            oprefix + "swarmboxenplot.no_borderline.svg",
            **c.figkws,
        )


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
        tifffile.imwrite(
            image.fname.replace(".tiff", ".fluo.tiff"), (h * 255).astype("uint8")
        )
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


def polygon_to_mask(
    polygon_vertices: tp.Sequence[tp.Sequence[float]],
    shape: tp.Tuple[int, int],
    including_edges: bool = True,
) -> Array:
    """
    Convert a set of vertices to a binary array.

    Adapted and extended from: https://stackoverflow.com/a/36759414/1469535.
    """
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.geometry.collection import GeometryCollection

    if including_edges:
        # This makes sure edge pixels are also positive
        grid = Polygon([(0, 0), (shape[0], 0), (shape[0], shape[1]), (0, shape[1])])
        poly = Polygon(polygon_vertices)
        if not poly.is_valid:
            poly = poly.buffer(0)
        inter = grid.intersection(poly)
        if isinstance(inter, (MultiPolygon, GeometryCollection)):
            return np.asarray([polygon_to_mask(x, shape) for x in inter.geoms]).sum(0) > 0
        inter_verts = np.asarray(inter.exterior.coords.xy).T.tolist()
    else:
        inter_verts = polygon_vertices
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T
    path = matplotlib.path.Path(inter_verts)
    grid = path.contains_points(points, radius=-1)
    return grid.reshape((shape[1], shape[0]))


if __name__ == "__main__" and "get_ipython" not in locals():
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit()
