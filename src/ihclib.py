"""
A package for the analysis of IHC data.
"""

from __future__ import annotations
import typing as tp
import json

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tifffile

from skimage.color import separate_stains, hdx_from_rgb
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from imc.types import Path, Array, DataFrame
from seaborn_extensions import swarmboxenplot

get_image_from_url: tp.Callable
upload_image: tp.Callable
get_urls: tp.Callable
get_population: tp.Callable
quantify_cell_intensity: tp.Callable

from imc.operations import get_population
from imc.operations import quantify_cell_intensity


def minmax_scale(x: Array) -> Array:
    return (x - x.min()) / (x.max() - x.min())


def normalize_illumination(image: Array) -> Array:
    import cv2

    # i = image / image.max()
    # eps = np.finfo(float).eps
    # i[i == 0] = eps
    # i = (i * (2 ** 16)).astype('uint16')
    i = image

    hh, ww = image.shape[:2]
    imax = max(hh, ww)

    # illumination normalize
    ycrcb = cv2.cvtColor(i, cv2.COLOR_RGB2YCrCb)

    # separate channels
    y, cr, cb = cv2.split(ycrcb)

    # get background which paper says (gaussian blur using standard deviation 5 pixel for 300x300 size image)
    # account for size of input vs 300
    sigma = int(5 * imax / 300)
    gaussian = cv2.GaussianBlur(y, (0, 0), sigma, sigma)

    # subtract background from Y channel
    y = y - gaussian + 100  #  + gaussian.max() - gaussian.min()

    # merge channels back
    ycrcb = cv2.merge([y, cr, cb])

    # convert to BGR
    q = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return q


def simplest_cb(img: Array, percent: float = 1) -> Array:
    """https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc"""
    import cv2

    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0),
    )
    for channel in cv2.split(img):
        cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0, 256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate(
            (
                np.zeros(low_cut),
                np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
                255 * np.ones(255 - high_cut),
            )
        )
        out_channels.append(cv2.LUT(channel, lut.astype("uint8")))
    return cv2.merge(out_channels)


metadata_dir: Path
data_dir: Path
results_dir: Path
phenotype_order: tp.List[str]
figkws: tp.Dict[str, tp.Any]
col: "ImageCollection"


class Image:
    def __init__(
        self,
        marker: str,
        image_file_name: Path,
        image_url: str = None,
        mask_file_name: Path = None,
        mask_url: str = None,
        normalize_illumination: bool = True,
    ):
        self.marker = marker
        self.image_file_name = image_file_name.absolute()
        self.image_url = image_url
        self.mask_file_name = (
            mask_file_name or self.image_file_name.replace_(".tif", ".stardist_mask.tiff")
        ).absolute()
        self.mask_url = mask_url
        self.col: ImageCollection = None
        self.normalize_illumination = normalize_illumination

    def __repr__(self):
        return f"Image of '{self.marker}': '{self.name}'"

    @property
    def name(self):
        return self.image_file_name.stem

    @property
    def image(self):
        try:
            img = tifffile.imread(self.image_file_name)
        except (FileNotFoundError, ValueError):
            img = get_image_from_url(self.image_url)

        if self.normalize_illumination:
            return normalize_illumination(img)
        return img

    @property
    def mask(self):
        try:
            return tifffile.imread(self.mask_file_name)
        except (FileNotFoundError, ValueError):
            return get_image_from_url(self.mask_url)

    @property
    def has_image(self):
        return self.image_file_name.exists()

    @property
    def has_mask(self):
        return self.mask_file_name.exists()

    def download(self, image_type: str = "image"):
        if image_type == "image":
            url = self.image_url
            file = self.image_file_name
        elif image_type == "mask":
            url = self.mask_url
            file = self.mask_file_name
        file.parent.mkdir()
        img = get_image_from_url(url, output_file=file)

    def segment(self, mode: str = "he") -> Array:
        if mode == "he":
            model = StarDist2D.from_pretrained("2D_versatile_he")
            model.thresholds = {"prob": 0.5, "nms": 0.3}
            mask, prob = model.predict_instances(normalize(self.image, 1, 99.8))
        elif mode == "fluo":
            model = StarDist2D.from_pretrained("2D_versatile_fluo")
            mask, prob = model.predict_instances(
                normalize(self.decompose_hdab()[0], 1, 99.8)
            )
        tifffile.imwrite(self.mask_file_name, mask)
        return mask

    def upload(self, image_type: str = "mask"):
        assert image_type == "mask", NotImplementedError(
            f"Uploading {image_type} is not yet implemented"
        )
        img_dict = self.col.files[self.marker][self.image_file_name.parts[-1]]
        uploaded = image_type in img_dict
        if self.has_mask and not uploaded:
            upload_image(
                self.mask,
                self.mask_file_name.parts[-1],
                subfolder_name=self.marker,
                subfolder_suffix="_masks" if image_type == "mask" else "",
            )

    def decompose_hdab(self, normalize: bool = False):
        ihc = np.moveaxis(separate_stains(self.image, hdx_from_rgb), -1, 0)
        if not normalize:
            return np.stack([ihc[0], ihc[1]])
        x = np.stack([minmax_scale(ihc[0]), minmax_scale(ihc[1])])
        return x

        # i = ihc.mean((1, 2)).argmax()
        # o = 0 if i == 1 else 1
        # x[i] = x[i] + x[o] * (x[o].mean() / x[i].mean())
        # hema = minmax_scale(x[0])
        # dab = minmax_scale(x[1])

        # fig, axes = plt.subplots(1, 4, sharex=True, sharey=True)
        # axes[0].imshow(self.image)
        # axes[1].imshow(ihc[..., 0], cmap=cmap_hema)
        # axes[2].imshow(ihc[..., 1], cmap=cmap_dab)
        # axes[3].imshow(ihc[..., 2])

        # hema = minmax_scale(ihc[0] / ihc.sum(0))
        # dab = minmax_scale(ihc[2] / ihc.sum(0))
        # hema2 = hema + dab * 0.33
        # dab2 = dab + hema * 0.33
        # hema = minmax_scale(hema2)
        # dab = minmax_scale(dab2)
        # return np.stack([dab, hema])

    def quantify(self, **kwargs):
        quant = quantify_cell_intensity(
            self.decompose_hdab(**kwargs), self.mask, border_objs=True
        )
        quant.columns = ["hematoxilyn", "diaminobenzidine"]
        quant.index.name = "cell_id"
        return quant.assign(image=self.name, marker=self.marker)


class ImageCollection:
    def __init__(
        self,
        files: tp.Dict[str, tp.Dict[str, tp.Dict[str, str]]] = {},
        images: tp.List[Image] = [],
    ):
        self.files = files
        self.images = images
        # self.files_json = metadata_dir / "ihc_files.box_dir.json"

        self.files_json = metadata_dir / "ihc_files.image_mask_urls.json"
        self.quant_file = data_dir / "quantification_hdab.csv"

        self.get_files(regenerate=False)
        self.generate_image_objs()

    def __repr__(self):
        return f"Image collection with {len(self.images)} images."

    @property
    def markers(self):
        return sorted(np.unique([i.marker for i in col.images]).tolist())

    def get_files(
        self,
        force_refresh: bool = False,
        exclude_keys: tp.List[str] = None,
        regenerate: bool = True,
    ):
        if exclude_keys is None:
            exclude_keys = []
        if force_refresh or not self.files_json.exists():
            files = get_urls()
            for key in exclude_keys:
                files.pop(key, None)
            json.dump(files, open(self.files_json, "w"), indent=4)
        self.files = json.load(open(self.files_json, "r"))

        if regenerate:
            return ImageCollection(files=self.files)

    def generate_image_objs(self, force_refresh: bool = False):
        images = list()

        if self.files is None:
            print("Getting file URLs")
            self.files = self.get_files()
        for sf in self.files:
            for name, urls in self.files[sf].items():
                image = Image(
                    marker=sf,
                    image_file_name=data_dir / sf / name,
                    image_url=urls["image"],
                    mask_url=urls.get("mask"),
                )
                image.col = self
                images.append(image)
        self.images = images

    def download_images(self, overwrite: bool = False):
        for image in tqdm(self.images):
            if overwrite or not image.has_image:
                image.download("image")

    def download_masks(self, overwrite: bool = False):
        for image in tqdm(self.images):
            if overwrite or not image.has_mask:
                image.download("mask")

    def upload_images(self):
        raise NotImplementedError
        for image in tqdm(self.images):
            ...

    def upload_masks(self, refresh_files: bool = True):
        for image in tqdm(self.images):
            image.upload("mask")

    def remove_images(self):
        for image in tqdm(self.images):
            image.image_file_name.unlink()

    def remove_masks(self):
        for image in tqdm(self.images):
            image.mask_file_name.unlink()

    def segment(self):
        model = StarDist2D.from_pretrained("2D_versatile_he")
        model.thresholds = {"prob": 0.5, "nms": 0.3}

        for image in tqdm(self.images):
            mask, _ = model.predict_instances(normalize(image.image, 1, 99.8))
            tifffile.imwrite(image.mask_file_name, mask)

    @property
    def quantification(self):
        if self.quant_file.exists():
            quants = pd.read_csv(self.quant_file, index_col=0)
            quants.index = quants.index.astype(int)
        else:
            quants = pd.DataFrame(
                index=pd.Series(name="cell_id", dtype=int),
                columns=["hematoxilyn", "diaminobenzidine", "image", "marker"],
            )
        return quants

    def quantify(
        self,
        force_refresh: bool = False,
        save: bool = True,
        transform_func: tp.Callable = None,
    ):
        # import multiprocessing

        # _quants = list()
        # for image in tqdm(images):
        #     q = image.quantify()
        #     q['hematoxilyn'] = transform_func(q['hematoxilyn'])
        #     q['diaminobenzidine'] = transform_func(q['diaminobenzidine'])
        #     _quants.append(q)
        # quants = pd.concat(_quants)

        quants = self.quantification
        _quants = list()
        for image in tqdm(self.images):
            e = quants.query(f"marker == '{image.marker}' & image == '{image.name}'")
            if e.empty or force_refresh:
                tqdm.write(image.name)
                q = image.quantify()
                if transform_func is not None:
                    q["hematoxilyn"] = transform_func(q["hematoxilyn"])
                    q["diaminobenzidine"] = transform_func(q["diaminobenzidine"])
                _quants.append(q)
        if force_refresh:
            quants = pd.concat(_quants)
        else:
            quants = pd.concat([quants] + _quants)
        if save:
            quants.to_csv(self.quant_file)
        return quants


def files_to_dataframe(files: tp.Dict[str, tp.Dict[str, str]]) -> DataFrame:
    """
    Convert the nested dict of image markers, IDS and URLs into a dataframe.
    """
    f = [pd.DataFrame(v).T.assign(marker=k) for k, v in files.items()]
    return (
        pd.concat(f)
        .reset_index()
        .rename(columns={"image": "image_url", "mask": "mask_url"})
    )


class Analysis:
    @staticmethod
    def plot_sample_image_numbers(df, value_type="intensity", prefix=""):
        # Illustrate number of samples and images for each marker and disease group
        group_var = "phenotypes"
        combs = [
            ("count", "phenotypes", "marker", "by_phenotypes"),
            ("count", "marker", "phenotypes", "by_marker"),
        ]
        for x, y, h, label in combs:
            fig, axes = plt.subplots(1, 2, figsize=(2 * 4, 1 * 4), sharey=True)
            # # samples per group
            p = (
                df.groupby(["marker", group_var])["sample_id"]
                .nunique()
                .rename("count")
                .reset_index()
            )
            # # images per group
            p2 = df.groupby(["marker", group_var]).size().rename("count").reset_index()
            for ax, df2, xlab in zip(axes, [p, p2], ["Unique samples", "Images"]):
                df2["phenotypes"] = pd.Categorical(
                    df2["phenotypes"], categories=phenotype_order, ordered=True
                )
                sns.barplot(
                    data=df2,
                    x=x,
                    y=y,
                    hue=h,
                    orient="horiz",
                    ax=ax,
                    palette=globals()[h[0] + "_palette"],
                )
                ax.set(xlabel=xlab)
            fig.savefig(
                results_dir / f"ihc.{prefix}{value_type}.images_{label}.svg",
                **figkws,
            )

    @staticmethod
    def plot_comparison_between_groups(df, value_type="intensity", prefix=""):
        # Compare marker expression across disease groups (DAB intensity)
        for y, hue in [("phenotypes", "marker"), ("marker", "phenotypes")]:
            pal = globals()[hue[0] + "_palette"]
            fig, axes = plt.subplots(1, 1, figsize=(4, 4))
            sns.barplot(
                data=df.reset_index(),
                x=q_var,
                y=y,
                orient="horiz",
                hue=hue,
                ax=axes,
                palette=pal,
            )
            fig.savefig(
                results_dir / f"ihc.{prefix}{value_type}.by_{y}.barplot.svg",
                **figkws,
            )

            fig, stats = swarmboxenplot(
                data=df.reset_index(),
                y=q_var,
                x=y,
                hue=hue,
                plot_kws=dict(palette=pal),
            )
            fig.savefig(
                results_dir / f"ihc.{prefix}{value_type}.by_{y}.swarmboxenplot.svg",
                **figkws,
            )
            # plot also separately
            for g in df.reset_index()[hue].unique():
                p = df.reset_index().query(f"{hue} == '{g}'")
                p["phenotypes"] = p["phenotypes"].cat.remove_unused_categories()
                fig, stats = swarmboxenplot(
                    data=p,
                    y=q_var,
                    x=y,
                    plot_kws=dict(palette=globals()[y[0] + "_palette"]),
                )
                fig.savefig(
                    results_dir
                    / f"ihc.{prefix}{value_type}.by_{hue}.{g}.swarmboxenplot.svg",
                    **figkws,
                )

    @staticmethod
    def plot_example_top_bottom_images(
        df, col, n: int = 2, value_type: str = "intensity", prefix=""
    ):
        # Exemplify images with most/least stain
        nrows = len(phenotype_order)
        ncols = 2 * 2

        def nlarg(x):
            return x.nlargest(n)

        def nsmal(x):
            return x.nsmallest(n)

        for marker in col.files.keys():
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
            for pheno, ax in zip(phenotype_order, axes):
                img_names = (
                    df.loc[marker]
                    .query(f"phenotypes == '{pheno}'")["diaminobenzidine"]
                    .agg([nsmal, nlarg])
                    .index
                )
                imgs = [
                    i
                    for n in img_names
                    for i in col.images
                    if i.name == n and i.marker == marker
                ]
                for a, img in zip(ax, imgs):
                    a.imshow(img.image)
                    a.set_xticks([])
                    a.set_yticks([])
                    a.set_xticklabels([])
                    a.set_yticklabels([])
                    v = df.loc[(marker, img.name), "diaminobenzidine"]
                    a.set(title=f"{img.name}\n{v:.2f}")
                ax[0].set_ylabel(pheno)

            fig.savefig(
                results_dir
                / f"ihc.{prefix}{value_type}_top-bottom_{n}_per_group.{marker}.svg",
                **figkws,
            )

    @staticmethod
    def plot_example_images(
        df,
        col,
        n: int = 3,
        value_type: str = "random",
        prefix="",
        orient: str = "landscape",
    ):
        comparts = ["airway", "vessel", "alveolar"]
        # Exemplify images with most/least stain
        if orient == "landscape":
            nrows = len(phenotype_order)
            ncols = len(comparts) * n
        elif orient == "portrait":
            ncols = len(phenotype_order)
            nrows = len(comparts) * n

        for marker in col.files.keys():
            output_file = (
                results_dir
                / f"ihc.{prefix}{value_type}_random_{n}_per_group.{marker}.{orient}.svg"
            )
            if output_file.exists():
                continue
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(ncols * 4, nrows * 4),
                gridspec_kw=dict(wspace=0, hspace=0.05),
            )
            if orient == "portrait":
                axes = axes.T
            for pheno, ax in zip(phenotype_order, axes):
                for i, compart in enumerate(comparts):
                    idx_names = (
                        df.loc[marker]
                        .query(f"phenotypes == '{pheno}'")["diaminobenzidine"]
                        .index
                    )
                    idx_names = [x for x in idx_names if compart in x]
                    img_names = np.random.choice(idx_names, min(n, len(idx_names)))
                    imgs = [
                        i
                        for n in img_names
                        for i in col.images
                        if i.name == n and i.marker == marker
                    ]
                    for a, img in zip(ax[i * n : (i + 1) * n], imgs):
                        a.imshow(img.image, rasterized=True)
                        a.set_xticks([])
                        a.set_yticks([])
                        a.set_xticklabels([])
                        a.set_yticklabels([])
                        v = df.loc[(marker, img.name), "diaminobenzidine"]
                        a.set(title=f"{img.name} - {v:.2f}")
                if orient == "landscape":
                    ax[0].set_ylabel(pheno)
                else:
                    ax[0].set_title(pheno)
            fig.savefig(output_file, **figkws)

    @staticmethod
    def gate_with_gmm_by_marker(
        df, values="diaminobenzidine", markers: tp.Sequence[str] = None
    ):
        if markers is None:
            markers = col.markers
        df["pos"] = np.nan
        for marker in markers:
            sel = df["marker"] == marker
            pos = get_population(df.loc[sel, values])
            df.loc[sel, "pos"] = pos
        return df

    @staticmethod
    def plot_gating(df, value_type="intensity", prefix=""):
        x, y = "hematoxilyn", "diaminobenzidine"
        fig, axes = plt.subplots(
            1,
            len(col.markers),
            figsize=(4 * len(col.markers), 4),
            sharex=True,
            sharey=True,
        )
        for ax, marker in zip(axes, col.markers):
            q = df.query(f"marker == '{marker}'")
            ax.axhline(0.3, linestyle="--", color="grey")
            ax.scatter(q[x], q[y], s=1, alpha=0.1, rasterized=True)
            ax.set(title=f"{marker}\n(n = {q.shape[0]:})", xlabel=x, ylabel=y)
            ax.scatter(
                q.loc[pos, x],
                q.loc[pos, y],
                s=2,
                alpha=0.1,
                rasterized=True,
                color="red",
            )
        fig.savefig(
            results_dir / f"ihc.{prefix}{value_type}.gating.by_marker.scatterplot.svg",
            **figkws,
        )

        # # plot also as histogram
        fig, axes = plt.subplots(
            1,
            len(col.markers),
            figsize=(4 * len(col.markers), 4),
            sharex=True,
            sharey=True,
        )
        for ax, marker in zip(axes, col.markers):
            q = df.query(f"marker == '{marker}'")
            ax.axhline(0.3, linestyle="--", color="grey")
            sns.distplot(q[y], kde=False, ax=ax)
            ax.set(
                title=f"{marker}\n(n = {q.shape[0]:,})",
                xlabel=x,
                ylabel=y,
            )
            sns.distplot(q.loc[q["pos"] == True, y], color="red", kde=False, ax=ax)
        fig.savefig(
            results_dir / f"ihc_image.{value_type}.gating.by_marker.histplot.svg",
            **figkws,
        )
