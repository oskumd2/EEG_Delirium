# Deep Learning-Based Intraoperative EEG Analysis for Postoperative Delirium Prediction

This is the official codebase for the paper "Deep Learning-Based Intraoperative EEG Analysis for Postoperative Delirium Prediction".

## Usage

Run `notebook_inference.ipynb` for Preprocessing, Model Training and Inference.

### Preprocessing
1. `.npy` files extraction of:
   - 20 EEG time series samples
   - Label
   - Patient ID
   - Surgery case ID
   - Age
   - Sex
   - Suppression ratio time series for each surgery case
2. Removing samples with corrupt (uniform) channel(s)
3. Adding Pink noise to the EEG time series included in development set, and normalizing every EEG time series (regardless of train, test)
4. Reducing the length of each time series to 2000

### Training Loop
1. Training each cross-fold of the development set and saving the train step size
2. Training the whole development set for the mean step size of above

### Inference
1. Comparison between model-based label and SR-based label based on Fisher's exact test
2. Evaluation and visualization for:
   - 1:1 undersampled
   - 2:1 undersampled
   - The original ratio test set

## Data Availability
CSV files containing patient information, model weight file (`final_model_full_training.pth`), and preprocessed `.npy` files are not available in this repository. Please contact the author (oskumd00@gmail.com) for more information.
