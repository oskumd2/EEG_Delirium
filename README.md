# A Deep Learning-based Prediction Model for Postoperative Delirium Using Intraoperative Electroencephalogram

This is the official codebase for the paper "A Deep Learning-based Prediction Model for Postoperative Delirium Using Intraoperative Electroencephalogram".

## Usage

Run `notebook_inference.ipynb` for Preprocessing, Model Training and Inference.

### EEG Preprocessing
1. `.npy` files extraction of:
   - 30 EEG time series samples
   - Label
   - Patient ID
   - Surgery case ID
   - Age
   - Sex
   - Suppression ratio time series for each surgery case
2. Removing samples with corrupt (uniform) channel(s)
3. Normalizing every EEG time series (regardless of train, test)
4. Reducing the length of each time series to 9600 (80Hz)
   
### DELPHI-EEG
1. Training each cross-fold of the development set and saving the train step size
2. Training the whole development set for the mean step size of above
3. Evaluation and visualization for:
   - 1:1 undersampled
   - 2:1 undersampled
   - The original ratio test set

### LR, ML
1. Training and Evaluation of Logistic Regression and ML models
 
### Posthoc Analysis
1. Interpretability Analysis Using EEG Spectral Power Correlations
2. Frequency Domain Perturbation Analysis

## Data Availability
CSV files containing patient information, model weight file (`final_model_full_training.pth`), and preprocessed `.npy` files are not available in this repository. Please contact the author (oskumd00@gmail.com) for more information.
