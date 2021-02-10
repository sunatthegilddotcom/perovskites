Use Cases:

 1. Loading image
	* If the border exists extract the region of interest  
	* Resize the image to standard dimensions
	* Set image to standard resolution
	* Send to processing step

 2. Processing image
	* Normalize pixel values
	* Enhance image contrast
 
 3. Loading metadata (operating conditions)
	* Import/parse json file
	* User input physical size of channel

 4. Loading csv data (Ld, time...)
	* 

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
	* Ld75 Time

	
