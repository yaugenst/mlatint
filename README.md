# ML@INT workshop 23.04.2024

## Run in Google Colab

You can simply open the notebooks in Colab via these links:
 - [notebooks/vae.ipynb](https://colab.research.google.com/github/yaugenst/mlatint/blob/master/notebooks/vae.ipynb)
 - [notebooks/fno.ipynb](https://colab.research.google.com/github/yaugenst/mlatint/blob/master/notebooks/fno.ipynb)
 - [notebooks/inverse_design.ipynb](https://colab.research.google.com/github/yaugenst/mlatint/blob/master/notebooks/inverse_design.ipynb)

## Run locally

Alternatively, you can run the notebooks locally, this should generally be faster.
I recommend setting up an empty Python 3.11 environment for this (e.g. via [conda](https://github.com/conda-forge/miniforge)) and cloning this repository:

```bash
conda create -n mlatint python=3.11 pip
conda activate mlatint
git clone https://github.com/yaugenst/mlatint
cd mlatint
pip install -e .
jupyter lab notebooks/
```

You can also `pip install` the package directly, but you'll have to download the notebooks separately then:
```bash
pip install git+https://github.com/yaugenst/mlatint
```
