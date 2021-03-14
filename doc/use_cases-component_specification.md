Use Cases:

1. Loading image
	* If the border exists extract the region of interest  
	* Resize the image to standard dimensions
	* Set image to standard resolution
	* Send to processing step

 2. Processing images
	* Normalize pixel values 
	* Enhance image contrast
 
 3. Loading metadata (operating conditions)
	* Import/parse json file
	* User input physical size of channel

 4. Training Autoencoder

 5. Neural Network 
	* Define layers
	* Train/Test
	* Output of array size defined by user

 6. Decide where to feed meta data
	* either into SUPER neural net?
		* beginning, middle?
	* into regression model 

 7. Regression model
	* Feed for regression model is neural network array and csv data
	
 8. Final ouput
	* Predicted Ld75 time

Component Specifications:

1. Loading/processing the image
	* Using "get_PL_paths" to get the locations of the tiff files
	* Using tifffile, import the image into the code
	* Converts image data to a numpy array
		* Maybe uses averaging to flatten timeseries data into a 2-D array
		* Or not if we find that the timeseries data is beneficial
	* Normalizes pixel values to range 0-1
	* Looks for sharp gradients in PL to determine where the edges of the channel are in the image
	* Crops the remaining image into a square

2. Loading json/metadata
	* Using the "pull_metadata_files", "pull search metadata", "get_metadata", and "get_metadata_df" functions in plva.py

3. Loading csv data
	* Using the "extract_raw_data" and "get_raw_data" functions in plva.py
	
4. Autoencoder
	* Dimensionally reduces the image size (from 32x32 to 8x8) using a convolution neural network
	* This is 

5. K-means/PCA
	* These further dimensionally reducts our image to find the most important groupings.

6. Multi Linear regression
	* After K-means/PCA has been completed, we send the results through a multi-linear regression moel
	* The features for this model include X, Y, Z... Need to add in later when we know
	* We predict Ld75 using this model 
