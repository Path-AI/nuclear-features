# pathai-nuclei: Deep-learning-based quantification of nuclear morphology

## Overview
> `pathai-nuclei` is the primary code repository for reproducing analyses in Abel et al. 2024: "Deep-learning quantified cell-type-specific nuclear morphology predicts genomic instability and prognosis in multiple cancer types"

You can read the full publication in [npj Precision Oncology]().

## Installation
- Clone this repo to your local machine using https://github.com/Path-AI/nuclear-features.git
- Estimated install time: <1 minute
- Estimated run time: varies between scripts (1-5 minutes)

## Contents
1. `/analysis` contains all Python and R code used to produce the results in the manuscript
2. `/data` contains all raw and cached data objects needed to reproduce the analysis, as well as to read in the training and evaluation data included in the manuscript.

## Version and package requirements 
To install the Python packages and dependencies needed to operate this code, please use Anaconda or Miniconda. From within this directory, do:
```shell
conda env create --name nuclei --file=environment.yml
conda activate nuclei
```

## Abstract 
While alterations in nucleus size, shape, and color are ubiquitous in cancer, comprehensive quantification of nuclear morphology across a whole-slide histologic image remains a challenge. 
Here, we describe the development of a pan-tissue, deep learning-based digital pathology pipeline for exhaustive nucleus detection, segmentation, and classification and the utility of this pipeline for nuclear morphologic biomarker discovery. 
Manually-collected nucleus annotations were used to train an object detection and segmentation model for identifying nuclei, which was deployed to segment nuclei in H&E-stained slides from the BRCA, LUAD, and PRAD TCGA cohorts. 
Interpretable features describing the shape, size, color, and texture of each nucleus were extracted from segmented nuclei and compared to measurements of genomic instability, gene expression, and prognosis. 
The nuclear segmentation and classification model trained herein performed comparably to previously reported models. 
Features extracted from the model revealed differences sufficient to distinguish between BRCA, LUAD, and PRAD. 
Furthermore, cancer cell nuclear area was associated with increased aneuploidy score and homologous recombination deficiency. 
In BRCA, increased fibroblast nuclear area was indicative of poor progression-free and overall survival and was associated with gene expression signatures related to extracellular matrix remodeling and anti-tumor immunity. 
Thus, we developed a powerful pan-tissue approach for nucleus segmentation and featurization, enabling the construction of predictive models and the identification of features linking nuclear morphology with clinically-relevant prognostic biomarkers across multiple cancer types.


## License
The contents of this repository are made available under CC BY-NC 4.0 (as provided in `license.txt`).
