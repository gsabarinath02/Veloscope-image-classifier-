
# EfficientNet-B0 Image Classification

This project trains and uses an **EfficientNet-B0** model to classify images as either **valid** or **invalid**. The repository contains scripts for training the model, running inference on a single image, and batch inference on a folder of images.

---

## **Contents**

1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Training the Model](#training-the-model)
4. [Running Inference on a Single Image](#running-inference-on-a-single-image)
5. [Running Inference on a Folder](#running-inference-on-a-folder)
6. [Folder Structure](#folder-structure)

---

## **Requirements**

Ensure you have the following installed:

- Python 3.8 or above
- PyTorch
- torchvision
- timm
- scikit-learn
- matplotlib
- seaborn
- PIL (Pillow)

You can install the required packages with:

```bash
pip install torch torchvision timm scikit-learn matplotlib seaborn pillow
```

---

## **Setup**

1. **Clone the repository or create a working directory:**
   ```bash
   mkdir image-classification
   cd image-classification
   ```

2. **Organize your dataset:**

   - Create a folder named `images` with the following structure:
     ```
     images/
     ├── valid_images/
     │   ├── valid_image1.jpg
     │   ├── valid_image2.jpg
     │   └── ...
     ├── invalid_images/
         ├── invalid_image1.jpg
         ├── invalid_image2.jpg
         └── ...
     ```
   - Place all **valid images** in the `valid_images` subfolder.
   - Place all **invalid images** in the `invalid_images` subfolder.

3. **Download the scripts:**

   - Save the training script as `train.py`.
   - Save the single-image inference script as `inference_single.py`.
   - Save the folder inference script as `inference_folder.py`.

---

## **Training the Model**

1. **Edit the `train.py` file:**
   - Ensure the `DATA_DIR` path points to your dataset folder (e.g., `/content/images` in Colab).

2. **Run the training script:**

   ```bash
   python train.py
   ```

   - The script will:
     - Train the model on the dataset.
     - Save the best-performing model to `best_efficientnet_b0.pth`.
     - Save the final model to `efficientnet_b0_final.pth`.

3. **Training Metrics:**
   - The script will display metrics such as training and validation loss, accuracy, and a confusion matrix.

---

## **Running Inference on a Single Image**

1. **Edit `inference_single.py`:**
   - Set the `MODEL_PATH` to the path of your saved model (e.g., `efficientnet_b0_final.pth`).
   - Set the `image_path` to the path of the image you want to classify.

2. **Run the script:**

   ```bash
   python inference_single.py
   ```

3. **Output:**
   - The script will output the prediction (`valid` or `invalid`) for the image.

   Example:
   ```
   Prediction for the image '/content/images/Valid_images/sample_image.jpg': valid
   ```

---

## **Running Inference on a Folder**

1. **Edit `inference_folder.py`:**
   - Set the `MODEL_PATH` to the path of your saved model (e.g., `efficientnet_b0_final.pth`).
   - Set the `folder_path` to the path of the folder containing images.

2. **Run the script:**

   ```bash
   python inference_folder.py
   ```

3. **Output:**
   - The script will output predictions for all images in the folder.

   Example:
   ```
   Predictions:
   Image: img1.jpg, Predicted Class: valid
   Image: img2.jpg, Predicted Class: invalid
   Image: img3.jpg, Predicted Class: valid
   ```

---

## **Folder Structure**

Ensure your project folder looks like this:

```
image-classification/
├── train.py              # Training script
├── inference_single.py   # Inference script for a single image
├── inference_folder.py   # Inference script for a folder of images
├── images/               # Dataset folder
│   ├── valid_images/     # Valid images
│   └── invalid_images/   # Invalid images
├── best_efficientnet_b0.pth  # Saved best model (after training)
└── efficientnet_b0_final.pth # Saved final model (after training)
```

---

## **Common Issues**

1. **No images found in the dataset:**
   - Ensure that your `images` folder is structured correctly with subfolders `valid_images` and `invalid_images`.

2. **CUDA not available:**
   - Ensure your system supports CUDA or set the `device` to `cpu` in the scripts.

3. **Unsupported file format:**
   - Only `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, and `.webp` files are supported.

---

## **Acknowledgments**

This project leverages:
- **EfficientNet-B0** from the [timm](https://github.com/rwightman/pytorch-image-models) library.
- PyTorch and torchvision for model training and data handling.

---

Feel free to customize or extend this README as needed!
