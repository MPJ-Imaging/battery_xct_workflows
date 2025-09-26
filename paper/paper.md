---
title: "Example Jupyter Notebooks for Quantitative Analysis of X-ray CT Data in Lithium-Ion Batteries"
tags:
  - batteries
  - x-ray computed tomography
  - image analysis
  - Jupyter notebooks
authors:
  - name: Matthew P. Jones
    orcid: 0000-0000-0000-0000        # replace
    affiliation: "1"
  - name: Huw C. W. Parks
    affiliation: "1"
  - name: Francesco Iacoviello
    affiliation: "1, 2, 3"
  - name: Rhodri Jervis
    affiliation: "1, 2, 3"
affiliations:
  - name: Electrochemical Innovation Lab, Department of Chemical Engineering, UCL, UK
    index: 1
  - name: The Faraday Institution, Harwell Science and Innovation Campus, Didcot, UK
    index: 2
  - name: Advanced Propulsion Lab, Marshgate, UCL, UK
    index: 3
date: 2025-09-26                      # update if needed
bibliography: paper.bib
---

# Summary

X-ray computed tomography (XCT) is a powerful tool for non-destructive characterization of lithium-ion batteries, enabling three-dimensional visualization of cell components and microstructural features. However, translating XCT images into quantitative insights often requires custom analysis workflows that are not openly available, limiting reproducibility and reuse.  

This submission provides a set of open Jupyter notebooks that demonstrate practical analysis workflows for battery XCT data. The notebooks focus on three tasks: (1) unrolling 2D slices of cylindrical “jelly roll” cells into polar coordinates (radial distance, angle), including fitting of an ideal spiral to quantify deviations from the expected geometry, (2) quantifying electrode overhang in cylindrical cells to assess manufacturing quality through multiple geometric measurements, and (3) analyzing gray-level variations in radial layers of cracked nickel–manganese–cobalt (NMC) particles, with the ability to process and compare a large number of particles to reveal population-level trends. Each notebook is accompanied by example data, including images and, where relevant, segmentation masks, so that the workflows can be executed without specialized preprocessing.  

The notebooks are implemented in Python and make use of widely adopted scientific libraries, including NumPy, SciPy, scikit-image, and Matplotlib. By focusing on clarity and reproducibility, they are intended both as ready-to-use analysis examples for battery researchers and as adaptable templates for the wider imaging science community. All notebooks and supporting data are archived with a Zenodo DOI to ensure long-term accessibility.  

By lowering the barrier to quantitative XCT analysis, these example workflows promote transparent and reproducible research in battery science, and they may also serve as a starting point for similar analyses in other domains of materials tomography.  

# Statement of need

X-ray computed tomography (XCT) is increasingly used in battery research to visualize cell components and degradation phenomena. While raw XCT data are widely generated, reproducible workflows for quantitative analysis remain scarce. Newcomers to the technique often struggle to move beyond qualitative analysis and many research groups rely on ad hoc scripts or proprietary software, which makes comparisons across studies difficult and slows the adoption of best practices.  

The example notebooks presented here address this gap by providing accessible, open, and well-documented workflows for XCT analysis tasks in batteries. The notebooks are designed for researchers who wish to:  

- Unroll cylindrical cell “jelly rolls” into polar coordinates (radial distance, angle) and fit an ideal spiral to evaluate deviations from the expected winding pattern,  
- Quantify electrode overhang in cylindrical cells to assess manufacturing quality through a range of geometric measures, and  
- Characterize and compare radial gray-level variations in large populations of cracked NMC particles, enabling statistical insights into structural heterogeneity.  

These workflows are intended both as *ready-to-use examples* for battery scientists and as *adaptable templates* for tomography researchers in other domains. By distributing the methods as Jupyter notebooks with example data, the project lowers the barrier for entry, encourages reuse, and promotes reproducibility in XCT-based research.  

# Illustrative outputs

![Unrolled jelly roll slice into polar coordinates with fitted spiral. 
Image generated using the example notebook and dataset [@zenodo2025software].](fig1.png)

![Radial gray-level profiles for a population of cracked NMC particles, 
demonstrating the ability to compare many particles statistically.](fig2.png)

# Acknowledgements

The author thanks colleagues and collaborators for feedback on the notebook design, and acknowledges the open-source Python scientific ecosystem (NumPy, SciPy, scikit-image, Matplotlib) that made this work possible.  

# References
