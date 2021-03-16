## Decoding spatial heterogeneities in Photoluminiscence images in Perovskite Thin films : <font color='purple'>**TEAM PEROVSKEET**</font>
[![Coverage Status](https://coveralls.io/repos/github/afarley9/perovskites/badge.svg?branch=main)](https://coveralls.io/github/afarley9/perovskites?branch=main)
[![Build Status](https://travis-ci.com/afarley9/perovskites.svg?branch=main)](https://travis-ci.com/afarley9/perovskites)


### Overview:

**Perovskite solar cells (PSCs)** have seen an increase in photo-conversion efficiencies from 17% to 25.2% in just a span of seven years. Although perovskite solar cells show promising efficiencies, their poor stabilities have restricted their commercial deployment. PSCs degrade quickly in the presence of oxygen, moisture, thermal stress and illumination, which makes predicting their payback lifetime a challenge for commercial manufacturers. This motivated the Hillhouse Group at the University of Washington, to develop a machine learning model which can predict the ***'lifetime'*** of a perovskite thin film using few early time series measurements and the environmental conditions of degradation. A well-trained model like this would allow the users to   
<img src="papers/figure_perovskites.png" alt="perovskites" width="500"/>


Currently, in the Hillhouse Group at University of Washington, 
![Routes](papers/figure1.png)


Develop a python package that can extract hidden information from spatial heterogeneities observed in PL images of perovskite thin films and quantify it to generate features for use in lifetime prediction models. Currently, PL image processing is primarily implemented in biological fields; we seek to restructure this technology for use in the material characterization of perovskite photovoltaics. This will provide a more robust picture of features which lead to degradation than current state of the art perovskite characterization. 
![Routes](papers/figure2.png)

### Potential Users:
* Other academic researchers in the thin-film PV community
* Research testbeds
* Solar-cell manufacturers

### Use Cases:
*Primary:*
* Predict lifetime of a perovskite (Ld80) for a perovskite based on early timeseries (spatial AIPL) data
* Enchance the rate of testing for thin-film PV
* Check for correlation between PL and diffusion length in perovskites

*Secondary:*
* Through correlating features to film lifetime, determine how to manufacture an optimally stable thin-film
* Extract spatial PL features from images to be fed into a linear regression model for Ld80 prediction
* Extract information for the PL images correlated to manufacturing processes

### Instructions:
* Check the [examples/](examples/) folder for introductory demonstrations on visualization, spatial analysis, etc.
* To use the data, first install [Git Large-File-Storage (LFS)](https://git-lfs.github.com). If you pull the repo and find that the `data/example.zip` is only of 1kB, try these two commands after `git pull origin main`- 
```
git lfs fetch
git lfs checkout
```

* However, for properly implementing the autoencoder, access to the larger PL datainfrastructure is required. To get access contact one of the contributors such that we can grant permissions. This will be necessary for the PL data loader to function properly.
* Utilize the .py files in the perovskites/utils folder to implement an array of different spatial analyzer tools (autoencoder, CNN, PCA, etc.).

### References
1. Early-time widefield absolute PL intensities and carrier diffusion length measurements can be used to predict the lifetime of perovskite thin films. But, this paper uses the mean pixel intensities of PL images, negleting the spatial heterogeneities [Stoddard et. al., ACS Energy Lett. 2020, 5, 3, 946–954](https://pubs.acs.org/doi/10.1021/acsenergylett.0c00164)
2. Macroscopic PL heterogeneities encode hidden information about variation in thin film preparation conditions, storage and consequent surface defect concentrations [Tze-Bin Song et al 2019 J. Phys. Energy 1 011002](https://iopscience.iop.org/article/10.1088/2515-7655/aaeee5)

### Datasets:
Hillhouse Lab Group Dataset

