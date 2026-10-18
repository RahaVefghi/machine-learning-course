# Machine Learning Coursework: Foundations & Practical Exercises

> 🎓 **Academic Portfolio:** This repository contains foundational exercises, algorithm implementations, and practical assignments developed during the **Machine Learning & Deep Learning Course**. More complex and specialized projects have been structured into dedicated standalone repositories linked below.

---

## 🌟 Featured Standalone Projects

The comprehensive, end-to-end applications developed during this coursework are hosted in separate repositories:

1. **[Captcha Digit Recognizer CNN](https://github.com/RahaVefghi/captcha-digit-recognizer-cnn)**
   - Segment and extract digits from noisy captchas using OpenCV contour morphology, classified with a regularized deep CNN.
2. **[Face & Gender Detection with MTCNN](https://github.com/RahaVefghi/face-gender-detection-mtcnn)**
   - Multi-task Cascaded Convolutional Networks (MTCNN) for face alignment and facial landmark localization, combined with an online SGDClassifier for real-time gender recognition.
3. **[Fire & Smoke Detection System](https://github.com/RahaVefghi/fire-detection-system)**
   - Vision-based hazard monitoring comparing distance-weighted KNN against deep neural networks with real-time visual alerting.
4. **[MNIST Multi-Algorithm Benchmark](https://github.com/RahaVefghi/mnist-algorithms-benchmark)**
   - A unified benchmark pipeline comparing 6 machine learning algorithms (Linear Regression, Logistic Regression, KNN+PCA, K-Means, MLP, CNN) on the standard MNIST digit dataset.

---

## 📁 Repository Structure

```text
machine-learning-course/
│
├── requirements.txt                      # Project dependencies
├── .gitignore                             # Files ignored by git
│
├── 01-classical-ml/                       # Supervised Machine Learning
│   ├── classification/                    # Classification algorithms
│   │   ├── iris_classification.py        # Iris flower classification with KNN
│   │   ├── diabetes_classification.py    # Diabetes diagnosis with missing value imputation & scaling
│   │   └── data/                          # Datasets (iris.data, diabetes.csv)
│   ├── regression/                        # Regression algorithms
│   │   ├── student_regression.py          # Linear regression: Study hours vs Exam scores
│   │   ├── petrol_regression.py           # Fuel consumption regression using SGDRegressor
│   │   └── data/                          # Datasets (student_scores.csv, petrol_consumption.csv)
│   └── preprocessing/                     # Feature engineering & encoding
│       ├── house_price_encoding.py        # Zipcode encoding via LabelBinarizer & price regression
│       ├── sklearn_encoding.py            # Comparison: OrdinalEncoder vs OneHotEncoder
│       └── data/                          # Dataset (HousesInfo.txt)
│
├── 02-unsupervised-learning/              # Unsupervised Learning & Clustering
│   ├── kmeans_basic.py                    # Fundamentals of K-Means on synthetic data
│   ├── kmeans_image.py                    # Clustering handwritten digits & cluster centroid visualization
│   ├── kmeans_mnist.py                    # K-Means clustering on MNIST digit representations
│   └── data/                              # Compressed MNIST data (mnist.npz)
│
├── 03-deep-learning/                      # Deep Neural Network Foundations
│   └── house-price-mlp/                   # Real estate price regression with Multilayer Perceptron
│       ├── house_price_mlp.py
│       └── data/
│
└── 04-computer-vision/                    # Computer Vision with OpenCV
    ├── opencv-basics/                     # Image basics, drawing primitives, color conversions
    │   ├── cv_basics.py                   # Geometric primitives and text rendering
    │   ├── opencv_ex2.ipynb               # Image filtering and color space notebook
    │   └── (Sample test images)
    └── face-detection/                    # MTCNN Face & Landmark detection basics
        ├── face_detection_mtcnn.py        # Facial bounding box & keypoint extraction
        └── maryam.jpg                     # Sample test image
```

---

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/RahaVefghi/machine-learning-course.git
cd machine-learning-course
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run sample scripts
```bash
# Iris flower classification
python 01-classical-ml/classification/iris_classification.py

# Student exam scores regression
python 01-classical-ml/regression/student_regression.py

# K-Means clustering on MNIST
python 02-unsupervised-learning/kmeans_mnist.py
```

---

## 🛠️ Technologies & Libraries
- **Python 3.10+**
- **Scikit-Learn:** KNN, LinearRegression, SGDClassifier, KMeans, StandardScaler, Pipeline
- **Pandas & NumPy:** Data wrangling, tabular manipulation, matrix computations
- **TensorFlow / Keras:** Multilayer Perceptron (MLP) architectures
- **OpenCV & MTCNN:** Computer vision, spatial operations, facial detection
- **Matplotlib:** Evaluation plots, confusion matrices, clustering visualizations
