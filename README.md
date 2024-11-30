# DETECTING-KERATOCONUS-EARLY-USING-MACHINE-LEARNING-KCN-

**Abstract** 

The study employed the possibility of the use of machine learning applications for the early detection of ocular disease that affects the eye. It seeks to detect the defects that are normal, present or suspicious. This was done using python software for testing the model which most accurately predicts the defect. Data used was obtained from https://www.kaggle.com/datasets/elmehdi12/keratoconus-detection. Data was preprocessed, with 80% used as testing data and 20% as training data. The simple model projection showed that the best performance in batches has an accuracy score of 68%. For the custom model, the best performance in batches has an accuracy score of 84%. This showed that the custom model performed better than the simple model. The study concluded that there is an opportunity to improve the model's generalising ability further if the epoch size is increased while retaining the callbacks setup. Though this would require more computational structure for better result
Keywords: Keratoconus, Machine Learning, Detection, Eye Condition

**BACKGROUND**

It is now understood that keratoconus is an eye condition that affects both eyes and is characterised by an uneven and unequal steepening of the cornea. This can lead to a reduction in visual acuity and irregular astigmatism. The thinning of the cornea usually occurs in the central or paracentral area, with the inferotemporal region being the most affected (SM et. al., 2021, Jacinto, et.al.,2022). The progressive cornea thinning caused by keratoconus can result in myopia, irregular astigmatism, and scarring, severely impacting a patient's quality of life. This disease can be debilitating and have significant effects on visual acuity.
Keratoconus progression causes vision loss and irregular astigmatism. However, linking data from developing technology to improve diagnostic precision is problematic. (Vazirani, et.al.,2013). The eye doctor must evaluate the information on diagnosis to establish the proper range of defects for each patient and base their clinical choice on the findings, which many times possess difficult (Donabedien, 2005).
The use of AI gives a chance for the development of creative solutions to improve difficulties encountered in the provision of real-time healthcare. Hence, this project seeks to investigate the challenge of using AI to identify keratoconus and to create an AI system for identifying Keratoconus situations that are normal, KCN, or suspicious.

**METHODS**

The directory contains information retrieved from Google Drive, mounted, and then imported into the Collab environment using Python. Numerous libraries and modules, including NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn, Tensorflow and Keras, were loaded into Python in order to analyse the data and create machine-learning models. The data was collected from three different centres, including one hospital and two clinics. 


**Reading the Data**	
The data, downloaded from Kaggle, consists of scanned images from corneal tomography, a non-invasive diagnostic imaging test that provides a 3D cornea map. This test is used to evaluate the cornea's shape, thickness, and curvature, which are essential for this analysis. The cornea is a transparent layer that covers the front of the eye and helps focus light to create a clear image of the retina. The shape and curvature of the cornea can affect how light is focused, and irregularities can lead to vision problems, including keratoconus. Doctors can diagnose and monitor keratoconus and other corneal conditions using corneal tomography to assess these factors. Two categories of people generate the images in the dataset:
volunteers who have had keratoconus diagnosed
volunteers with good vision
The dataset was loaded using the Keras API of the TensorFlow framework, which provides opportunities for optimisation like batch loading, shuffling, and automated image resizing. The dataset was also divided into training and validation sets to avoid overfitting.

**Exploratory Data Analysis**

The dataset's images contain 3D data, so proper data structuring is required to make it ideal for machine learning modelling. The two categories that make up this phase are data standardisation and data performance configuration. The process of rescaling data from an RGB structure, where pixel values range from 0 to 255, to a [0, 1] format is known as data standardisation. Standardisation is required to ensure that the input data is consistent and that the model can effectively learn from it.
Caching and prefetching streamlined model training data loading during data performance configuration. Caching commonly used data in memory saves loading time. Prefetching loads the next batch of data into memory before processing the current batch. This strategy keeps the model processing data while minimising load time.
Standardising the data format and streamlining the data loading procedure is necessary to properly structure 3D data in image datasets to enhance model performance. 2961 image files comprised the training data, of which 2369 were used for training and 592 for validation across two different models—1050 image files made up the test data, which was used to assess the model's performance. The custom model was trained for 20 epochs, while the first model was trained for 10 epochs with additional callbacks to prevent overfitting. The models were trained in batches of 32 images per iteration.

**Data Pre-Processing**

The batch size for processing image data was 32, and the image size was 90x90. The shuffle parameter was set to true for randomisation, and the dataset was divided into 80% training and 20% validation sets. The label model used integer encoding by default and a random seed of 43 for reproducibility, but the test data was categorical. The training and validation data processing was separated using the direction parameter during the training phase.
To expedite training, the Dataset.cache method loads data from the disk during the first epoch and keeps it in memory for subsequent epochs. To reduce I/O latency and boost throughput, the Dataset.prefetch method overlaps data preprocessing and model execution. These techniques can be effectively applied to various datasets using a function, creating a pipeline for prefetching and caching operations during training. Faster training times and improved model performance are the results of this.
Testing the model's performance on fresh, untested data is crucial to the success of the use case. It should be noted, though, that the test images must adhere to the format of corneal tomography scans, particularly imaging methods used to produce 3D cornea maps. It is essential to ensure that the test images used for evaluation represent this type of data because the trained model was created to analyse and interpret this type of image data. Failure to do so would lead to incorrect predictions and make the solution useless when it was put into practice.

**Model Development** 

A simple model and a customised model are both used in this methodology. Several layers, including Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, and Dropout, are used in both models. They also compute the model's performance using the Adam optimiser, the SparseCategoricalCrossentropy loss function, and the Accuracy metric. Both models also employ the ReduceLROnPlateau and EarlyStopping callbacks for monitoring, adjusting, and stopping training as needed.

Appendix I shows the training and validation loss is plotted over multiple training epochs, and while the training loss is declining, the validation loss is not. This suggests that the model is overfitting to the training data. When a model becomes overly complicated and learns the patterns in the training data too thoroughly, it begins to fit the noise in the data. This is known as overfitting. Because of this, the model does well with training data but poorly with test data. Several strategies, including regularisation, dropout, and early stopping, can be used to prevent overfitting.

The plot in Appendix II demonstrates that adding more training epochs can enhance model performance and raises the possibility of overfitting. The model will be trained for 20 epochs to test its performance due to computational constraints. However, the ideal number of epochs may vary depending on the dataset and problem being addressed, so it's crucial to test various values to find the right balance.
Model testing is repetitious, time-consuming, and may not increase performance. Hyperparameter tuning optimises hyperparameters for a model architecture and dataset, can improve model performance and prevent overfitting. In this case, the bespoke model outperformed the simple model, suggesting that manual fine-tuning may have benefited if the correct values were set. The Keras Tuner, which can automatically look for the ideal set of hyperparameters, can be used to tune the hyperparameters to achieve further gains.

The risk of overfitting was one issue with the custom model's initial training, which was resolved using callbacks like early stopping and reducing the learning rate on plateau. To avoid overfitting, hyperparameter tuning can also assist in determining the best mix of regularisation strategies and additional hyperparameters. Hyperparameter tuning is an effective method for enhancing the functionality of machine learning models, resulting in better generalisation of new data.
The model architecture is defined in the search space for each hyperparameter during hyperparameter tuning. This search space can be changed by subclassing the Keras Tuner API's HyperModel class or using a model builder function. For computer vision applications, HyperXception and HyperResNet are two pre-defined HyperModel classes. In this case, a customised model builder function is used, enabling an inline definition of hyperparameters like learning rate, batch size, activation functions, and regularisation strategies. A crucial step in hyperparameter tuning is defining the hyper model, which establishes the range of hyperparameters to be investigated for optimum performance.

The tuning recommendations are disregarded if hyperparameter adjustment did not result in appreciable gains in the model's performance over the original values used in the custom model. This could indicate that the search space was poorly defined, the tuning procedure was not sufficiently thorough, or the original parameters already captured the issue's essence. However, it is crucial to properly plan the search space and tuning process, monitor the model's performance, and consider other improvement options, including changing the model architecture, incorporating more data, or utilising different optimisation approaches. In conclusion, looking at additional possibilities for enhancing the model's performance is reasonable if hyper-tuning does not result in appreciable gains.

**RESULTS & CONCLUSIONS**

The result prediction function was done, and the simple model’s average performance was 53.6, while the customised model’s average was 61.65. the model was previewed to see the best performance along the batch iteration of the test data. The function checks batch prediction then the best accurate prediction is selected before the predicted and actual plots are compared. From the simple model projection, it was seen that the best performance in batches has an accuracy score of 68%; see Appendix III. For the custom model, the best performance in batches has an accuracy score of 84% (See Appendix IV).
From the multiple model testing, it was observed that:
The repeated task of model testing has some benefits and setbacks, which is why hyper-parameter tuning has gained popularity in the history of machine learning and AI. After multiple model testing, the custom model's performance yielded a better result when compared to the simple model used initially.
This can be attributed to the manual method of fine-tuning the model, which was tested using different values repeatedly. KERAS TUNER will be applied to the custom model to discover if there would be an improvement in the model's predictive power.
One notable observation was that the higher the epochs on the custom model, the better the model trained. However, this could result in overfitting of the model, which was why callbacks were implemented to checkmates overfitting, but let's see if hyper-tuning might help in this case.

**MODEL HYPER-PARAMETER TUNINGS STEPS**

In building a model for hyper-tuning, the hyperparameter search space will be defined in addition to the model architecture (following the guide from tensorflow).
The hyper model (the model for hyper tuning) can be defined using two approaches.
By using a model builder function
By subclassing the HyperModel class of the Keras Tuner API
Tensorflow has two pre-defined HyperModel classes - HyperXception and HyperResNet for computer vision applications, which can also be used; a custom version will be used for now. This model builder class will return a compiled model and uses hyperparameters defined inline to hyper-tune the model.
After multiple epochs, runtime ranging from the random searching to the hyper model retraining, the model hyper tuning did not perform better than the initial parameters used in the custom model.
Therefore, the tuning suggestions would be ignored.




**CONCLUSION AND RECOMMENDATION**

The whole process revealed that deep learning has so much still to offer; developing a solution for detecting Keratonocus is an innovation that will boost the effect of ML and AI in the health sector.

In conclusion

The process of loading the dataset and carrying out EDA was well implemented.

Shuffle parameters helped avoid the model overfitting the data during the data loading phase.

The model-building phase showed that even a simple CNN model could achieve up to 70% accuracy on new test cases.

The custom model showed a better performance than the simple model and also provided an opportunity to improve the model's generalising ability further if the epochs size is increased while retaining the callbacks setup. However, this would require more computational structure for better results.

The hyper-parameter fine-tuning phase did not provide solutions in terms of the parameter settings to be used for the custom model.

**The following Recommendations were made:**

To further improve this work, there is a need for more training data.

The batch size affected the model's performance, so further studies must be conducted to ascertain the most suitable batch size for this analysis (when using a very robust computation system).

This solution used two basic model architectures; for further work, there is a need to try other complex architectures, such as the vgg16 model, which has robust architecture.

The choice of accuracy as a metric is to determine the number of cases correctly predicted; in addition to this metric, other metrics can also be utilised.

Also, future solutions should implement a shuffled data loading as it helps to avoid overfitting of data.



**REFERENCES**

Donabedian, A. 2005. Evaluating the quality of medical care. 1966. Milbank Q.;83(4):691-729. Doi: 10.1111/j.1468-0009.2005.00397.x. PMID: 16279964; PMCID: PMC2690293.
Jacinto S. et. al., 2022. Keratoconus: An updated review, Contact Lens and Anterior Eye45
Ng, S. M., Ren, M., Lindsley, K. B., Hawkins, B. S., Kuo, I. C.2021. Transepithelial versus epithelium-off corneal crosslinking for progressive keratoconus. Cochrane Database Syst Rev. 3(3): CD013512. Doi: 10.1002/14651858.CD013512.pub2. PMID: 33765359; PMCID: PMC8094622.
Vazirani J, and Basu S.2013.  Keratoconus: current perspectives. Clin Ophthalmol. 2013;7:2019-30. Doi: 10.2147/OPTH.S50119. Epub, PMID: 24143069; PMCID: PMC3798205.







APPENDIX I



APPENDIX II











APPENDIX III

APPENDIX IV


