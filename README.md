# Battery XCT Notebooks  

A collection of open Jupyter notebooks demonstrating **quantitative workflows for X-ray computed tomography (XCT) data in lithium-ion batteries**.  

These notebooks provide reproducible examples for:  
1. **Electrode overhang assessment** - measuring overhang features in cylindrical cells to assess manufacturing quality.  
2. **Cylindrical cell unrolling** - transforming 2D slices of cylindrical cells into polar coordinates (radial distance, angle) and fitting an ideal spiral to quantify deviations from the expected geometry.
3. **Cracked particle analysis** - analyzing and comparing radial gray-level profiles in populations of cracked NMC particles, enabling population-level insights.  

All notebooks include lightweight example datasets (cropped, 8-bit volumes) so they run quickly and out of the box.  

---

## Installation  

These notebooks require **Python ≥3.9**.  

1. Clone the repository:  
   ```bash
   git clone https://github.com/MPJ-Imaging/battery_xct_notebooks.git
   cd battery_xct_notebooks
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
   - `01_overhangs_analysis.ipynb`  
   - `02_cylindrical_cell_unrolling.ipynb`  
   - `03_cracking_active_particles.ipynb`  

3. Run all cells to reproduce the workflows. Each notebook generates example plots and figures.  

---

## Data availability  

- This repo includes **lightweight volumes and images** (by X-ray CT standards) in `data/` so notebooks run out of the box.  
- Larger original datasets are not necessary for reproducing the examples.  
- Figures in the accompanying paper are generated from these reduced datasets.

---

## License  

This project is licensed under the terms of the [MIT License](LICENSE).  

