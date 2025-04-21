<!-- Notebook Icon Header -->
<p align="center">
  <img src="https://img.icons8.com/fluency/96/artificial-intelligence.png" alt="AI Icon" width="70" style="margin-right: 10px;"/>
  <img src="https://img.icons8.com/emoji/96/sun-emoji.png" alt="Sun Icon" width="70"/>
</p>

<h1 align="center">🌞 SolarStormNet</h1>
<h3 align="center">Multimodal Deep Learning for Solar Storm Classification</h3>

<p align="center">
  <em>Powered by CNN · ResNet50V2 · MobileNetV2 · Late Fusion Ensemble</em>
</p>

![redundancy-resiliency-banner-1680x885](https://github.com/user-attachments/assets/15cb7aa0-27b1-41a7-9474-5f64b9fb354c)

**SolarStormNet** is a robust deep learning pipeline designed to classify solar storm imagery using a multimodal dataset consisting of *continuum* and *magnetogram* images. It integrates handcrafted CNN architectures and transfer learning models (ResNet50V2 and MobileNetV2) into an ensemble system that captures rich spatial and modality-aware features. The final hybrid model improves classification accuracy for different magnetic classes: `alpha`, `beta`, and `betax`.

---

## 🧠 Objective

The goal of this project is to automatically classify solar active regions using AI models trained on solar image data. These classifications can help scientists and engineers better understand solar activity and predict potentially hazardous space weather events.

---

## 📁 Dataset Overview

**Source**: [Kaggle - Solar Storm Recognition Dataset](https://www.kaggle.com/datasets/djzezev/solar-storm-recognition-dataset)  
**Modality Types**:
- 🌕 **Continuum**: White-light imagery
- 🧲 **Magnetogram**: Magnetic field mapping

**Magnetic Classes**:
- `alpha`: Simple unipolar sunspots
- `beta`: Bipolar group with a simple configuration
- `betax`: Complex or undefined magnetic configuration

**Directory Structure**:
```
solar-storm-recognition-dataset/
└── project1/
    ├── training/
    │   ├── continuum/
    │   │   ├── alpha/
    │   │   ├── beta/
    │   │   └── betax/
    │   └── magnetogram/
    │       ├── alpha/
    │       ├── beta/
    │       └── betax/
    └── testing/
        ├── continuum/
        └── magnetogram/
```

---

## 🔍 Project Structure

The notebook is structured in the following steps:

1. **Library Imports**  
   Includes `TensorFlow`, `Keras`, `Matplotlib`, `Seaborn`, `PIL`, `OpenDatasets`, etc.

2. **Dataset Analysis & Visualization**  
   - Automatically scans folders and counts images by class/modality.
   - Displays bar plots and sample images from each category.

3. **Data Preparation**  
   - Loads image paths and labels.
   - Splits training/validation sets.
   - Applies real-time data augmentation (rotation, flip, shift, etc.).

4. **Modeling**
   - `Custom CNN`: A compact CNN built from scratch.
   - `ResNet50V2`: A deeper model pretrained on ImageNet.
   - `MobileNetV2`: Lightweight architecture optimized for speed.
   - All models use a softmax head for 3-class classification.

5. **Training**  
   - Uses callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.
   - Trains each model independently and saves the best weights.

6. **Hybrid Model Construction**  
   - Combines feature outputs from each model.
   - Uses concatenated features in a final dense classifier.
   - Jointly predicts the class from multi-model outputs.

7. **Evaluation**
   - Plots training/validation loss and accuracy.
   - Outputs confusion matrix and classification report.
   - Runs final prediction on test dataset.
---
# 📊 Model Evaluation and Visual Results

---

## 🖼️ Misclassified and Correctly Classified Samples

Below are examples of true and predicted classifications using the trained hybrid model. The fifth image shows a misclassified case (`True: 0`, `Pred: 1`).

## Misclassification Examples
![output2](https://github.com/user-attachments/assets/1c1abe7c-55bd-4f44-89bc-06823f26ff59)

---

## 📉 Training Loss Comparison

This plot compares the validation loss over epochs for the CNN, ResNet50V2, and MobileNetV2 models. The CNN model shows the steepest decline, indicating faster convergence.
## Loss Comparison
![LOSS](https://github.com/user-attachments/assets/ec401155-7ec3-418c-926b-822a4c08e381)

---

## 📈 Accuracy Over Epochs

The following plot illustrates the accuracy progression for all models across training epochs:
## CNN, ResNet50V2, MobileNetV2 Classifiers
![2](https://github.com/user-attachments/assets/1472fda1-a3b9-4f5b-8ce9-c25570c32acf)

## Hybrid Classifier
![output3](https://github.com/user-attachments/assets/16154279-d9cf-4872-ab3c-7259d086e98a)

---

## 📋 Evaluation Metrics Summary

| Model              | Loss   | Modality Accuracy | Magnetic Class Accuracy |
|-------------------|--------|-------------------|--------------------------|
| **CNN**           | 0.2006 | 93.24%            | 99.98%                   |
| **ResNet50V2**     | 0.2680 | 89.63%            | 99.98%                   |
| **MobileNetV2**    | 0.2936 | 88.35%            | 99.98%                   |
| **Hybrid (Val)**   | 0.1344 | 95.78%            | 99.98%                   |
| **Hybrid (Test)**  | 0.6188 | 82.94%            | 100.00%                  |

---

## 🧠 Interpretation

- The **hybrid model** achieved the best **validation performance**, suggesting that multimodal fusion enhances classification accuracy.
- **CNN** outperforms the larger pretrained models on this specific dataset, likely due to dataset size and domain difference from ImageNet.
- All models perform exceptionally well in predicting the magnetic class with near-perfect accuracy, indicating robustness in detecting solar region polarity.


---
## 🧬 Techniques Used

- **ImageDataGenerator**: Real-time augmentation and preprocessing
- **Transfer Learning**: Adapting pretrained models to solar storm imagery
- **Hybrid Ensembling**: Late fusion strategy using feature concatenation
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam with dynamic learning rate scheduling

---

## 📦 Installation

Install all required dependencies using pip:

```bash
pip install kaggle opendatasets tqdm tensorflow matplotlib seaborn pandas pillow
```

---

## 🚀 How to Use

1.**Download Dataset**:
```python
import opendatasets as od
od.download('https://www.kaggle.com/datasets/djzezev/solar-storm-recognition-dataset')
```

2.**Set Dataset Path**:
```
python
BASE_DIR = "solar-storm-recognition-dataset/project1/project1"
```

3.**Run the Notebook**:
- Step through the cells from data loading to model evaluation.
- Train CNN, ResNet, MobileNet models.
- Execute hybrid ensemble block.
- Evaluate on validation and test sets.

---

## 📈 Results & Performance

- **Training Accuracy**: >90% on hybrid ensemble model
- **Validation Accuracy**: Stable after fine-tuning with early stopping
- **Confusion Matrix**: Shows high precision and recall for all classes
- **Ensemble Model**: Demonstrated improved generalization compared to individual models

---

## 🧠 Future Work

- Implement attention-based fusion for modality-aware feature learning
- Incorporate temporal modeling (e.g., ConvLSTM) for solar flare forecasting
- Deploy model as an inference API using FastAPI or TensorFlow Serving
- Visualize learned filters and class activation maps (CAM)

---

## 🤝 Credits

- **Dataset**: DjZeZeV via Kaggle
- **Libraries**: TensorFlow, Keras, OpenDatasets, Matplotlib, Seaborn
- **Team**: Deep Learning Engineers at Galala University (or your team name)

---

## 📄 License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this code for both academic and commercial purposes.
