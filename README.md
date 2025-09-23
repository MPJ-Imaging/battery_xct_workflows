# Battery XCT Notebooks  

A collection of open Jupyter notebooks demonstrating **quantitative workflows for X-ray computed tomography (XCT) data in lithium-ion batteries**.  

These notebooks provide reproducible examples for:  
1. **Jelly roll unrolling** - transforming 2D slices of cylindrical cells into polar coordinates (radial distance, angle) and fitting an ideal spiral to quantify deviations from the expected geometry.  
2. **Electrode overhang assessment** - measuring overhang features in cylindrical cells to assess manufacturing quality.  
3. **Cracked particle analysis** - analyzing and comparing radial gray-level profiles in large populations of cracked NMC particles, enabling population-level insights.  

All notebooks include lightweight example datasets (cropped, 8-bit volumes) so they run quickly and out of the box.  

---

## Installation  

These notebooks require **Python ≥3.9**.  

1. Clone the repository:  
   ```bash
   git clone https://github.com/MPJ-Imaging/battery_xct_notebooks.git
   cd battery-xct-example-notebooks
2. Create a fresh environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # (Linux/macOS)
   venv\Scripts\activate      # (Windows)
3. Install Dependencies:
   ```bash
   pip install -r requirements.txt
4. Interactive Widget:
   Some notebooks use ipywidgets for parameter exploration. These are tested with Jupyter Notebook ≥7.0 (or JupyterLab ≥4.0).

## Usage  

1. Launch Jupyter Notebook:  
   ```bash
   jupyter notebook
   ```
or
   Launch Jupyter Lab:
   ```bash
   jupyter lab
   ```
2. Open the notebooks in the `notebooks/` folder:  
   - `01_jellyroll_unroll_spiral.ipynb`  
   - `02_overhang_quality.ipynb`  
   - `03_cracked_nmc_radial_graylevels.ipynb`  

3. Run all cells to reproduce the workflows. Each notebook generates example plots and figures.  

---

## Data availability  

- This repo includes **lightweight cropped 8-bit volumes** in `data/` so notebooks run out of the box.  
- Larger original datasets are not necessary for reproducing the examples.  
- Figures in the accompanying paper are generated from these reduced datasets.
- Where Machine Learning Segmentation or De-noising is utilised this is stated, but models are not made available. 

---

## Citation  

If you use these notebooks, please cite the Zenodo archive:  

> Jones, M. P. (2025). *Battery XCT Example Notebooks* (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX  

---

## License  

This project is licensed under the terms of the [MIT License](LICENSE).  

