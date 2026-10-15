# Satellite Image Classifier

Satellite image classification with fastai transfer learning (ResNet18), plus a comparison of five CNN architectures, a CPU vs GPU timing study and a learning rate finder experiment.

## Problem

Given a satellite image, predict its land-cover class. The model predicts one of 4 classes:

- `cloudy`
- `desert`
- `green_area`
- `water`

Each class is split 80/20 into train and test folders. The training folder is then split again 80/20 into training and validation (3,604 / 900 images).

## Notebooks

All notebooks are written in Spanish.

| # | Notebook | What it does | Open |
|---|---|---|---|
| 1 | [`E2_Principal.ipynb`](E2_Principal.ipynb) | Builds the train/test split, fine-tunes a ResNet18 with fastai, evaluates it on the test set, cleans the dataset with `ImageClassifierCleaner`, tests the model on five external images, pushes it to the Hugging Face Hub and links the Gradio app. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/santiagodie/satellite-image-classifier/blob/main/E2_Principal.ipynb) |
| 2 | [`E2_Adicional1_Arquitecturas.ipynb`](E2_Adicional1_Arquitecturas.ipynb) | Trains ResNet18, ResNet34, EfficientNet-B0, MobileNetV3 Small and ConvNeXt Tiny (fastai + timm) under the same conditions and compares accuracy, loss, training time and size. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/santiagodie/satellite-image-classifier/blob/main/E2_Adicional1_Arquitecturas.ipynb) |
| 3 | [`E2_Adicional2_cpu_vs_gpu.ipynb`](E2_Adicional2_cpu_vs_gpu.ipynb) | Trains the same ResNet18 for 3 epochs on CPU and on GPU and compares training and inference times. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/santiagodie/satellite-image-classifier/blob/main/E2_Adicional2_cpu_vs_gpu.ipynb) |
| 4 | [`E2_Adicional3_LR_finder.ipynb`](E2_Adicional3_LR_finder.ipynb) | Runs fastai's `lr_find` and compares the suggested learning rate (valley) with 1e-3 and 1e-2. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/santiagodie/satellite-image-classifier/blob/main/E2_Adicional3_LR_finder.ipynb) |

## Key results

All figures come from the saved notebook outputs.

**Main model** (`E2_Principal.ipynb`, ResNet18, `fine_tune(5)`, trained on CPU):
- Validation accuracy was 1.000 at the last epoch.
- On the held-out test set, `learn.validate()` gives loss 0.0051 and accuracy **0.9982** (error rate 0.0018).
- Before the test DataBlock was normalized with ImageNet statistics, test accuracy was only 0.312.

**External images:** the model was tried on five images taken from outside the dataset (Google Earth and the web). It got 2 of 5 right: the two `desert` images. It predicted `desert` for the `green_area`, `water` and `cloudy` images. The model does not generalize well to images from other sources.

**Architecture comparison** (`E2_Adicional1_Arquitecturas.ipynb`, 5 epochs, Colab T4 GPU, validation split):

| Architecture | Accuracy | Val loss | Time (s) | Params (M) |
|---|---|---|---|---|
| resnet34 | **1.0000** | 0.0008 | 148.1 | 21.81 |
| convnext_tiny | **1.0000** | 0.0009 | 382.6 | 28.61 |
| resnet18 | 0.9989 | 0.0029 | 107.8 | 11.71 |
| mobilenetv3_small_100 | 0.9722 | 137.1902 | 104.6 | 2.11 |
| efficientnet_b0 | 0.9711 | 0.1089 | 152.3 | 5.33 |

ResNet34 and ConvNeXt Tiny tie on accuracy. ResNet34 trains about 2.6× faster and has 7 million fewer parameters.

**CPU vs GPU** (`E2_Adicional2_cpu_vs_gpu.ipynb`, ResNet18, 3 epochs):

| Device | Training time (s) | Validation accuracy | Inference (ms/image) |
|---|---|---|---|
| CPU | 2639.33 | 0.9967 | 67.28 |
| GPU (T4) | 67.31 | 0.9967 | 2.29 |

The GPU trains about 39× faster and runs inference about 29× faster, with the same accuracy.

**LR finder** (`E2_Adicional3_LR_finder.ipynb`, ResNet18, 5 epochs):

| Learning rate | Accuracy | Val loss | Time (s) |
|---|---|---|---|
| 2.51e-03 (suggested valley) | 1.0000 | **0.0015** | 125.2 |
| 1.00e-03 | 0.9989 | 0.0037 | 115.8 |
| 1.00e-02 | 1.0000 | 0.0020 | 121.9 |

The suggested learning rate gives the lowest validation loss, although all three reach about 99.9–100% accuracy.

## Tech stack

Python · fastai · PyTorch · timm · scikit-learn · pandas · Matplotlib · Hugging Face Hub · Gradio · Google Colab

## How to run

Open a notebook in Google Colab with the badges above. A GPU runtime is recommended; the CPU run takes about 44 minutes for 3 epochs. These are the install commands as they appear in the notebooks (not re-verified):

```bash
pip install fastai -Uq                     # E2_Principal.ipynb, E2_Adicional2_cpu_vs_gpu.ipynb
pip install fastai timm -Uq                # E2_Adicional1_Arquitecturas.ipynb, E2_Adicional3_LR_finder.ipynb
pip install huggingface_hub["fastai"] -Uq  # E2_Principal.ipynb, only needed to push the model to the Hub
```

No versions are pinned. `E2_Principal.ipynb` imports `google.colab`, so it is meant to run in Colab.

The notebooks download a zipped copy of the dataset (`archive.zip`), the exported model (`satellite_model.pkl`) and the five external test images with `wget`. Those URLs have been replaced by the placeholder `DATA_URL`. To run the notebooks, download the dataset from Kaggle (see [Data](#data)) and provide `archive.zip` with one folder per class (`cloudy`, `desert`, `green_area`, `water`). The external test images are not included.

## Data

[Satellite Image Classification](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification) is a public dataset on Kaggle by `mahmoudreda55`. The Kaggle page lists the license as "Data files © Original Authors". The data is not included in this repository.

## Model

The ResNet18 classifier from `E2_Principal.ipynb` is available on the Hugging Face Hub: [`santiagodie/satellite`](https://huggingface.co/santiagodie/satellite) (uploaded with `push_to_hub_fastai`).

A Gradio app for the model was deployed on Hugging Face Spaces, but free CPU hardware is no longer available for this account.

## Known limitations

- **The dataset was cleaned, but the model was not retrained afterwards.** `ImageClassifierCleaner` removed 14 training images (7 `cloudy`, 4 `green_area`, 3 `water`). The notebook says the model should be trained again, but no training cell follows. The exported `satellite_model.pkl` and the model on the Hub were both trained before the cleaning.
- Near-perfect validation and test scores come from images of the same source. On five external images, accuracy drops to 2/5, with a strong bias towards `desert`.
- In `E2_Principal.ipynb`, the markdown summary of the test results says "99.7%" accuracy, but the output is 0.99823 (99.8%).
- In `E2_Adicional2_cpu_vs_gpu.ipynb`, the CPU and GPU runs call `fine_tune` on the same learner object and were evidently run in separate runtime sessions. The comparison table is typed in by hand from the printed outputs, so running the notebook top to bottom does not reproduce both measurements.
- The architecture, CPU/GPU and LR finder comparisons report validation accuracy on a random 20% split of the training folder, not on the held-out test set. Each configuration was run only once.
- MobileNetV3 Small reaches 97.2% accuracy but has a validation loss of 137.19, with spikes up to 1747.8 during training, which points to unstable training.
- The best architecture in the comparison (ResNet34) was not used for the published model, which is still the ResNet18 from the main notebook.

---

Built for the Artificial Intelligence course at the University of La Rioja (2025–26).
