# Visual Question Answering (VQA) Implementation

This repository contains the implementation of a Visual Question Answering (VQA) system based on the "Show, Attend and Tell" framework. The system combines computer vision and natural language processing using attention mechanisms to answer questions related to input images.

## Requirements
1. **Dataset**:
   - You will need to download the [Flickr8k Dataset](https://forms.illinois.edu/sec/1713398) or any other dataset of your choice.
   - Place all images in a folder named `images` in the root directory of this repository.
   - Include the `captions_fixed.txt` file in the same directory as your images.


2. **Folder Structure**:
The repository structure is as follows:
Final project
    - data
        - images
        - captions_fixed.txt
    - evaluate.py
    - train.py
    - decoder.py
    - encoder.py
    - dataset.py

3. **Run Training**:
   - To train the model, use the following command:
     ```bash
     python train.py
     ```
   - This will generate encoder.pth and decoder.pth
 
4. **Evaluate the Model**:
   - After training, evaluate the model and generate predictions:
     ```bash
     python evaluate.py
     ```
   - The results will be saved in `result.txt`.
