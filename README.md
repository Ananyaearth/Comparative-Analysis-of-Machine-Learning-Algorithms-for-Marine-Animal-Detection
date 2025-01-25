# Comparative-Analysis-of-Machine-Learning-Algorithms-for-Marine-Animal-Detection

Marine animal classification is a crucial task in ecological research, biodiversity monitoring, and conservation. This project explores machine learning techniques to classify marine animals into five categories: Dolphin, Fish, Lobster, Octopus, and Sea Horse, addressing challenges like underwater distortions and lighting variations.

## Project Overview

Underwater environments present unique challenges for classification tasks due to distortions and inconsistent lighting conditions. This project compares the performance of five machine learning models:

- **Random Forest (RF)**
- **Support Vector Machines (SVM)**
- **K-Means Clustering**
- **K-Nearest Neighbors (KNN)**
- **Convolutional Neural Networks (CNN)**

Through rigorous preprocessing and data augmentation techniques, this project aims to improve model robustness and accuracy.

### Key Contributions
1. Introduced advanced preprocessing techniques, including normalization, resizing, and data augmentation, to enhance model generalization.
2. Leveraged both traditional machine learning methods and deep learning approaches.
3. Achieved an accuracy of **92%** with CNNs, surpassing the previous benchmark of 91.94%.

---

## Authors 
- Ananya Verma
- Manav Khambhayata
- Kanha Tayal
- Smarak Patnaik
- Annirudha Singh
---

## Dataset

The dataset used in this project includes:
- **Training images**: 1241
- **Validation images**: 250
- **Test images**: 100
- **Categories**: Dolphin, Fish, Lobster, Octopus, Sea Horse
Kaggle Dataset link
The dataset was preprocessed using:
- **Normalization**: To standardize pixel intensity values.
- **Resizing**: Ensuring consistent input dimensions for all models.
- **Augmentation**: Techniques such as rotation, flipping, and brightness adjustments to increase dataset diversity.

The dataset used in this project is available on Kaggle:

- **Marine Image Dataset for Classification**: [Link to Dataset](https://www.kaggle.com/datasets/ananya12verma/marine-image-dataset-for-classification)
---

## Methods

### Traditional Machine Learning Models
1. **Random Forest (RF)**: Ensemble method using decision trees for classification.
2. **Support Vector Machines (SVM)**: Used for finding the optimal hyperplane.
3. **K-Means Clustering**: Unsupervised learning for pattern recognition.
4. **K-Nearest Neighbors (KNN)**: Instance-based learning for classification.

### Deep Learning Model
5. **Convolutional Neural Networks (CNN)**:
   - Designed for deep feature extraction.
   - Pre-trained **VGG16** model used for comparison but failed to capture complex underwater patterns effectively.

### Preprocessing Techniques
- **Principal Component Analysis (PCA)**: For dimensionality reduction in traditional models.
- **Data Augmentation**: Enhanced CNN robustness against distortions.

---

## Results

| Model                  | Accuracy (%) |
|------------------------|--------------|
| Random Forest (RF)     | 75.00        |
| Support Vector Machines (SVM) | 87.00        |
| K-Means Clustering     |             |
| K-Nearest Neighbors (KNN) | 72.00        |
| Convolutional Neural Networks (CNN) | **92.00**   |

### Key Insights
- CNNs excelled in feature learning, particularly for complex patterns in underwater images.
- Traditional methods like RF and SVM struggled with high environmental variability.
- Pre-trained VGG16 models were not effective for this dataset.

---

## Conclusion

This project demonstrates the effectiveness of CNNs in marine animal classification tasks, achieving state-of-the-art performance with an accuracy of 92%. The results underline the potential of deep learning approaches to address real-world ecological challenges.

---

## Usage

### Requirements
- Python 3.x
- TensorFlow
- scikit-learn
- OpenCV
- NumPy
- Matplotlib

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/marine-animal-classification.git
   cd marine-animal-classification

