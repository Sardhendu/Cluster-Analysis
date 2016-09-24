This code showcases a robust and efficient way of clustering categorical data.

1. Order-Purchase: 
	1. crt_dataset_models: This module contains all the important codework including building the similarity matrix for categorical attributes. The module contains code for finding the best centroids to avoid random choice and uses k-means in an efficient way for faster retrieval. 

	2. feature_selection: This module uses the concept of imformation gain to understand which feature are more valuable. However this module doesnt play a role in the process (used only for analyzing).

	3. main: This modules creates/stores the dataset, builds/stores the model and stores the output clusters into the disk for further analysis.   



Please note that the code was only used for analysing the algorithm with different dataset taken form UCI repository, the code is not in its most optimized form.