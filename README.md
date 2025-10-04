# Battery XCT Workflows  

A collection of open Jupyter notebooks demonstrating **quantitative quality assessment workflows for X-ray computed tomography (XCT)** data of lithium-ion cylindrical cells.  

These notebooks provide reproducible examples for:  
1. **Electrode overhang assessment** - measuring overhang features in cylindrical cells to assess manufacturing quality.
2. **Cylindrical canister assessment** - measuring diameter, wall thickness, eccentricity and detecting denting in the canister of a battery.
3. **Cylindrical cell electrode winding assessment** - transforming 2D slices of cylindrical cells into polar coordinates (radial distance, angle) and fitting an ideal spiral to quantify deviations from the expected geometry.

All notebooks include lightweight example datasets (cropped, 8-bit volumes) so they run quickly and out of the box.  

---

## Cloud Hosting

These notebooks can be run in **binder** or **google colab** (links below). Simply run all cells to reproduce the workflows. Each notebook generates example plots and figures.

- `01_cylindrical_cell_overhangs.ipynb`  
  [![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MPJ-Imaging/battery_xct_workflows/HEAD?labpath=notebooks/01_cylindrical_cell_overhangs.ipynb)  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MPJ-Imaging/battery_xct_workflows/blob/HEAD/notebooks/01_cylindrical_cell_overhangs.ipynb)  

- `02_cylindrical_cell_can.ipynb`  
  [![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MPJ-Imaging/battery_xct_workflows/HEAD?labpath=notebooks/02_cylindrical_cell_can.ipynb)  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MPJ-Imaging/battery_xct_workflows/blob/HEAD/notebooks/02_cylindrical_cell_can.ipynb)  

- `03_cylindrical_cell_electrode_winding.ipynb`  
  [![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MPJ-Imaging/battery_xct_workflows/HEAD?labpath=notebooks/03_cylindrical_cell_electrode_winding.ipynb)  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MPJ-Imaging/battery_xct_workflows/blob/HEAD/notebooks/03_cylindrical_cell_electrode_winding.ipynb)  

---

## Installation  

These notebooks require **Python ≥3.9**. Follow steps below to run locally.

1. Clone the repository:  
   ```bash
   git clone https://github.com/MPJ-Imaging/battery_xct_workflows.git
   cd battery_xct_workflows
2. Create a fresh environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # (Linux/macOS)
   venv\Scripts\activate      # (Windows)
3. Install Dependencies:
   ```bash
   pip install -r requirements.txt
4. Note on Widgets:
   The notebooks use ipywidgets for parameter exploration. These are tested with Jupyter Notebook ≥7.0 (or JupyterLab ≥4.0). On older version you may need to manually enable widgets.
   ```bash
   pip install ipywidgets
   jupyter nbextension enable --py widgetsnbextension

## Usage  

1. Launch Jupyter Notebook:  
   ```bash
   jupyter notebook
   ```
   or Launch Jupyter Lab:
   ```bash
   jupyter lab
   ```
2. Open the notebooks in the `notebooks/` folder:  
3. Run all cells to reproduce the workflows. Each notebook generates example plots and figures.

---

## Data availability  

- This repo includes **lightweight volumes and images** (by X-ray CT standards) in `data/` so notebooks run out of the box.  
- Larger original datasets are not necessary for reproducing the examples.  
- Figures in the accompanying paper are generated from these reduced datasets.

---

## License  

This project is licensed under the terms of the [MIT License](LICENSE).  

