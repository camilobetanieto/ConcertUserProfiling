# User Profiling at a Music Festival from Attendees’ Mobility Data

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Data Science MSc thesis

## Project Organization

The actual thesis document is contained in `reports`.

To examine or run the code, everything is organized and displayed in notebooks, found in the the `notebooks/` folder in order (e.g., `0-preprocessing-timetables.ipynb`, `1-preprocessing-fix-polygons-h3pandas.ipynb`, ...), and organized in subfolders corresponding to `preprocessing/` and `analysis/`. The source code is in `attendee_profiling`
The `envs/` folder contains the `.yml` files with the required dependencies. All notebooks can run with the `skmob_updated_h3` environment, except when it is explicitly stated in the notebook. 

```
├── LICENSE                <- MIT license
├── README.md              <- The top-level README for developers using this project.
│
├── notebooks              <- Jupyter notebooks. Organized in subfolders for analysis and preprocessing,
│                             an labeled in sequential order.
│
├── reports                <- Generated outputs and final thesis document.
│   |── CBN_MS_Thesis.pdf  <- Final thesis document.
│   └── figures            <- Generated graphics and figures.
│
├── envs                   <- The .yml files with the requirements.
│
│
└── attendee_profiling     <- Source code for use in this project.
    │
    ├── __init__.py             <- Marks this as a Python package
    │
    ├── config.py               <- Useful variables and configuration
    │
    .
    .
    .
```

--------

