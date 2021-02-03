## Decoding spatial heterogeneities in Photoluminiscence images in Perovskite Thin films

### Goals:
Develop a package that can extract hidden information from spatial heterogeneities observed in PL images of perovskite thin films and quantify it to generate features for use in lifetime prediction models.

1. Create a python package that implements image-processing techniques to pre-process widefield PL image data and extract features to be used in a ML model.
2. Using a CNN on widefield AIPL images directly to predict lifetime of a perovskite (Ld75) for a perovskite and also check for correlation between PL and diffusion length in perovskites. 3. Using an autoencoder to extract spatial PL features from images to be fed into a linear regression model for Ld75 prediction.

### References
1. Early-time widefield absolute PL intensities and carrier diffusion length measurements can be used to predict the lifetime of perovskite thin films. But, this paper uses the mean pixel intensities of PL images, negleting the spatial heterogeneities [Stoddard et. al., ACS Energy Lett. 2020, 5, 3, 946–954](https://pubs.acs.org/doi/10.1021/acsenergylett.0c00164)
2. Macroscopic PL heterogeneities encode hidden information about variation in thin film preparation conditions, storage and consequent surface defect concentrations [Tze-Bin Song et al 2019 J. Phys. Energy 1 011002](https://iopscience.iop.org/article/10.1088/2515-7655/aaeee5)
