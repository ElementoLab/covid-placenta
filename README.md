# IHC analysis of placental tissue from COVID-19 patients

[![Zenodo badge](https://zenodo.org/badge/doi/___doi1___.svg)](https://doi.org/___doi1___)
[![medRxiv badge](https://zenodo.org/badge/doi/__doi1___.svg)](https://doi.org/__doi1___) ⬅️ read the preprint here


**SARS-CoV-2 Infects Syncytiotrophoblast and Activates Inflammatory Responses in the Placenta**

Argueta et al. 2021


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

Feel free to use some virtualization or compartimentalization software such as virtual environments or conda to install the requirements.
