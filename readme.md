# This is the repository for the Aneja Lab AAA Segmentation Project

All model training was performed via nnUNet which can be found [here](https://github.com/MIC-DKFZ/nnUNet/tree/master)

## Code Flow
- Preprocessing tasks are designed to be run individually and as needed
- Normalization is intended to standardize the image color chanels and should be run before slice trimming
- Slice trimming is designed to isolate the portion of the scan that has a segmentation present. It creates a new mask and a new scan with only the slices that were segmented. This replicates a user choosing the upper and lower bounds to be segmented
- Metrics can be run as needed. Each of them are designed to be run on an inference output folder that contains the scan, the manual segmentation and the model generated segmentation
