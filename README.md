
# Image Process App using MATLAB

## Overview

This is an Image Processing App developed in MATLAB, offering a range of functionalities for image enhancement and noise manipulation. The app includes various filters, noise generation functions, and image enhancement techniques.

## Features

### 1. Filters

- **Mean (Average) Filtering:** Smooths images using the mean of pixel values within a specified kernel.
- **Median Filtering:** Removes impulsive noise by replacing pixel values with the median within a specified kernel.
- **Adaptive Median Filtering:** Dynamically adjusts the filter size based on local pixel intensity.

### 2. Noise Generation

- **Additive Uniform Noise:** Adds uniform noise to the input image with user-defined parameters.
- **Additive Gaussian Noise:** Adds Gaussian noise to the input image with user-defined parameters.
- **Additive Salt & Pepper Noise:** Introduces salt and pepper noise to the input image with user-defined parameters.
- **Additive LogNormal Noise:** Adds log-normal distributed noise to the input image with user-defined parameters.
- **Additive Rayleigh Noise:** Adds Rayleigh distributed noise to the input image with user-defined parameters.
- **Additive Exponential Noise:** Introduces exponential noise to the input image with user-defined parameters.
- **Additive Erlang Noise:** Adds Erlang (gamma) distributed noise to the input image with user-defined parameters.

### 3. Image Enhancement

- **Histogram Equalization:** Enhances image contrast by equalizing the histogram.
- **Gaussian Filtering:** Applies Gaussian smoothing to the image using a specified standard deviation.

## Usage

### Prerequisites

- MATLAB installed on your machine.

### Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/image-processing.git
   cd image-processing
2. Open MATLAB and navigate to the project directory.
3. Run the imageProcess.m file.
