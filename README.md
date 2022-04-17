# IHC analysis of placental tissue in COVID-19

[![Zenodo badge](https://zenodo.org/badge/doi/10.5281/zenodo.5182825.svg)](https://doi.org/10.5281/zenodo.5182825) <- dataset repository

**Inflammatory Responses in the Placenta upon SARS-CoV-2 Infection Late in Pregnancy - analysis of IHC data**
Lissenya B. Argueta*, Lauretta A. Lacko*, Yaron Bram*, Takuya Tada, Lucia Carrau, André Figueiredo Rendeiro, Tuo Zhang, Skyler Uhl, Brienne C. Lubor, Vasuretha Chandar, Cristianel Gil, Wei Zhang, Brittany Dodson, Jeroen Bastiaans, Malavika Prabhu, Sean Houghton, David Redmond, Christine M. Salvatore, Yawei J. Yang, Olivier Elemento, Rebecca N. Baergen, Benjamin R. tenOever, Nathaniel R. Landau, Shuibing Chen, Robert E. Schwartz, Heidi Stuhlmann. iScience 2022. [doi:10.1016/j.isci.2022.104223](https://doi.org/10.1016/j.isci.2022.104223)


## Organization

- The [metadata](metadata) directory contains metadata relevant the samples and acquired data
- The [src](src) directory contains source code used to analyze the data
- Raw data should be under the `data` directory after download from Zenodo.
- Outputs from the analysis will be present in a `results` directory.


## Reproducibility

### Running

To see all available steps type:
```bash
$ make
```

To reproduce analysis using the pre-preocessed data, one would so:

```bash
$ make help
$ make requirements   # install python requirements using pip
$ make download_data  # download processed from Zenodo
$ make analysis       # run the analysis scripts
```

#### Requirements

- Python 3.7+ (was run on 3.9.2)
- Python packages as specified in the [requirements file](requirements.txt) - install with `make requirements` or `pip install -r requirements.txt`. It is paramount to have `scikit-image>=0.18.2`!
