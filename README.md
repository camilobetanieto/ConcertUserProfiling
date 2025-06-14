# User Profiling at a Music Festival from Attendees’ Mobility Data

Data Science MSc thesis on spatiotemporal analysis of human mobility at a music festival using anonymized Wi-Fi traces.

## Project Organization

The final thesis document can be found at `reports/CBN_MS_Thesis.pdf`.

To explore or reproduce the code and analysis, Jupyter notebooks are provided in the `notebooks/` directory. These are:
- Labeled in sequential order (e.g., `0-preprocessing-timetables.ipynb`, `1-preprocessing-fix-polygons-h3pandas.ipynb`, ...)
- Organized into subfolders (`preprocessing/`, `analysis/`) based on their function

The main source code used by the notebooks is in the `attendee_profiling/` directory.

The `envs/` folder contains `.yml` files defining Conda environments. Unless otherwise noted in a specific notebook, all notebooks run with the `skmob_updated_h3` environment.

```
├── LICENSE                <- MIT license
├── README.md              <- Project overview and structure.
│
├── notebooks              <- Jupyter notebooks. Organized in subfolders for analysis and preprocessing
│                             and labeled in sequential order.
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

## Abstract

Clustering techniques are used in different domains to uncover patterns that would otherwise remain hidden in complex, unstructured datasets, supporting practical applications such as customer profiling, social behavior analysis, and traffic flow optimization. However, these methods are not typically tailored to the study of human mobility in events held at designated locations, such as music festivals and conventions, where the combination of activities and spatio-temporal dynamics introduces additional layers of complexity to the analysis. This study presents customized preprocessing and trajectory clustering adaptations for interpreting anonymous Wi-Fi traces of event attendees, addressing context-specific challenges such as unidentified sources of signals, uneven sample rates, and noisy sequences, all within an unsupervised setting with no ground truth. This strategy is compared with a network science approach that applies community detection to bipartite graphs built from implicit attendee feedback, discussing the distinct perspectives offered by each method. The clustering methods identified groups of attendees who stayed primarily within the two main audience zones and others with more exploratory behavior. Among the exploratory participants, some followed consistent movement patterns between venue areas, while others exhibited more irregular trajectories. These clusters, which also reflect musical preferences, provided different insights from those of the community detection techniques, which identified smaller groups with a stronger focus on music preference. These findings contribute to the understanding of participant behavior and present a methodological approach adaptable to similar event settings.


--------

