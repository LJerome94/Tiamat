<h1 align="center"> Tiamat </h1>
<p align="center">
This repository was made as a project for an undergraduate scientific computing class.
It contains classes, functions and scripts for studying the Mandelbrot set in the context of chaos theory.
</p>
<p align="center">
<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
</p>

# Requirements

All the requirements are present in `./pyproject.toml`. To set up this repo, I recommand you to install [Poetry](https://python-poetry.org/) and run

```bash
poetry install
```

in the main directory.

# File structure

Here is a tree of the important directories:

```
.
├── data/
├── figures/
├── main.py
└── tiamat/
```

- `./data/`: This folder holds the outputs of the simulations.
- `./figures/`: This contains the figures create from `./data/` and `./make_plots.py`.
- `./main.py`: This is the main script that runs the computations from the parameters in `./params.yml`.
- `./tiamat`: The module that defines the required class and functions.

# How to use

In order to perform simluations, write its (or their) parameter(s) in `./params.yml` an then run

```bash
python main.py
```

To then produce figures from the results, specify which data to plot in `./make_plot.py` and run

```bash
python make_plots.py
```


**Trivia:** In the spirit of chaos theory, this repo is named after the Mesopotamian diety of chaos.