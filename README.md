# Anomaly Detection with VLMs

Repository provides the source code for the project on Anomaly Detection with Vision Language Models

### [Slides]()

## Overview

The project supports two VLM backends: CLIP and GEM. Two techniques are implemented to enhance the accuracy of anomaly detection: multi-scale feature extraction, bank of faetures. The system can:

- Process images of industrial objects
- Perform image processing techniques on query images
- Detect and localize anomalies using a vision expert

## Repository Structure

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Sassanmtr/ADVLM.git
cd ADVLM
```

2. Install dependencies:
```bash
conda create --name advlm_env python=3.9
conda activate advlm_env
pip install -r requirements.txt
```

3. Download the MVTec-AD dataset and rename it to mvtec and place it in the root folder of the repository

## Usage
