# Pathology Image Processing Pipeline

This project implements a step-by-step pipeline for processing histopathological images. It is designed to handle data preprocessing, model training, and downstream tasks such as gland segmentation and detection.

## Project Overview

The main goal of this project is to process pathological images to accurately segment and detect glands. The pipeline is structured in the following stages:

1. **Data Preprocessing**  
   - Load and clean histopathology images.
   - Normalize and augment data to improve model generalization.

2. **Gland Segmentation**  
   - Utilize **YOLO** (You Only Look Once) for segmenting gland regions from the images.
   - Generate masks that highlight glandular structures.

3. **Gland Detection**  
   - Apply **U-Net** for precise detection and delineation of individual glands within the segmented regions.
   - Produce detailed masks for downstream analysis.

## Models Used

- **YOLO** – Efficiently detects and segments gland regions in large histopathology images.  
- **U-Net** – Performs fine-grained detection and segmentation on previously identified gland areas.

## Workflow

```mermaid
graph TD
A[Raw Histopathology Images] --> B[Data Preprocessing]
B --> C[YOLO Segmentation]
C --> D[U-Net Detection]
D --> E[Final Segmentation & Detection Masks]
