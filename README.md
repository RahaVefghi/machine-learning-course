# Machine Learning Course Exercises & Assignments
### مجموعه تمرین‌ها و تکالیف عملی دوره یادگیری ماشین و یادگیری عمیق

> 🎓 **Academic Portfolio:** این ریپازیتوری شامل تمرین‌های پایه‌ای، پیاده‌سازی الگوریتم‌ها و تکالیف هفتگی دوره یادگیری ماشین است. پروژه‌های پیشرفته‌تر و بزرگ‌تر دوره به صورت ریپازیتوری‌های مستقل توسعه داده شده‌اند که لینک آن‌ها در ادامه قرار دارد.

---

## 🌟 پروژه‌های برجسته و مستقل دوره (Featured Standalone Repositories)

پروژه‌های چندمرحله‌ای و کاربردی دوره در قالب مخازن اختصاصی زیر قرار دارند:

1. **[Captcha Digit Recognizer with CNN](https://github.com/)**
   - تفکیک و استخراج کانتور ارقام کپچا با فیلترهای مورفولوژی OpenCV و طبقه‌بندی با شبکه عمیق CNN و Data Augmentation.
2. **[Face & Gender Detection with MTCNN](https://github.com/)**
   - استخراج چهره و لندمارک‌های صورت با MTCNN و طبقه‌بندی جنسیت با SGDClassifier و نمایش بلادرنگ Bounding Box.
3. **[Fire & Smoke Detection System](https://github.com/)**
   - سیستم پایش تصویری خطر آتش‌سوزی با مقایسه KNN و شبکه‌های عصبی عمیق (MLP) به همراه رابط استنتاج تصویری.
4. **[MNIST Multi-Algorithm Benchmark](https://github.com/)**
   - بنچمارک مقایسه‌ای همزمان ۶ الگوریتم (Linear Regression, Logistic Regression, KNN+PCA, KMeans, MLP, CNN) روی دیتاست MNIST.

---

## 📁 ساختار این مخزن (Repository Structure)

```text
machine-learning-course/
│
├── requirements.txt                      # کتابخانه‌های مورد نیاز
├── .gitignore                             # فایل‌های نادیده گرفته‌شده توسط گیت
│
├── 01-classical-ml/                       # یادگیری ماشین نظارت‌شده (Supervised Learning)
│   ├── classification/                    # الگوریتم‌های دسته‌بندی
│   │   ├── iris_classification.py        # دسته‌بندی گل زنبق با KNN
│   │   ├── diabetes_classification.py    # پیش‌بینی دیابت با مدیریت مقادیر تهی و استانداردسازی
│   │   └── data/                          # دیتاست‌های کم‌حجم (iris.data, diabetes.csv)
│   ├── regression/                        # الگوریتم‌های رگرسیون
│   │   ├── student_regression.py          # رگرسیون خطی رابطه ساعات مطالعه و نمره
│   │   ├── petrol_regression.py           # رگرسیون مصرف سوخت با SGDRegressor
│   │   └── data/                          # دیتای student_scores.csv و petrol_consumption.csv
│   └── preprocessing/                     # تکنیک‌های مهندسی ویژگی و پیش‌پردازش
│       ├── house_price_encoding.py        # انکودینگ کد پستی با LabelBinarizer و رگرسیون قیمت مسکن
│       ├── sklearn_encoding.py            # مقایسه OrdinalEncoder و OneHotEncoder
│       └── data/                          # دیتای HousesInfo.txt
│
├── 02-unsupervised-learning/              # یادگیری بدون نظارت و خوشه‌بندی
│   ├── kmeans_basic.py                    # مفاهیم پایه خوشه‌بندی K-Means روی داده‌های سنتزی
│   ├── kmeans_image.py                    # خوشه‌بندی ارقام Sklearn و مصورسازی مراکز خوشه‌ها
│   ├── kmeans_mnist.py                    # خوشه‌بندی تصاویر ارقام MNIST با K-Means
│   └── data/                              # دیتای فشرده mnist.npz
│
├── 03-deep-learning/                      # مبانی شبکه‌های عصبی (Deep Learning)
│   └── house-price-mlp/                   # پیش‌بینی قیمت مسکن با شبکه عصبی چندلایه (Dense + SGD)
│       ├── house_price_mlp.py
│       └── data/
│
└── 04-computer-vision/                    # پردازش تصویر با OpenCV
    ├── opencv-basics/                     # مبانی تصویر، اشکال هندسی و تبدیلات
    │   ├── cv_basics.py                   # رسم مستطیل، دایره و متن روی تصویر
    │   ├── opencv_ex2.ipynb               # نوت‌بوک تمرین‌های فیلترگذاری و تبدیل‌های رنگی
    │   └── (تصاویر نمونه)
    └── face-detection/                    # مقدمات کار با مدل MTCNN
        ├── face_detection_mtcnn.py        # تشخیص چهره و استخراج نقاط کلیدی چشم و بینی
        └── maryam.jpg                     # تصویر تست
```

---

## ⚙️ راهنمای اجرا (Installation & Setup)

```bash
# کلون مخزن
git clone https://github.com/<YOUR_USERNAME>/machine-learning-course.git
cd machine-learning-course

# نصب پکیج‌ها
pip install -r requirements.txt

# نمونه اجرای تمرین دسته‌بندی زنبق:
python 01-classical-ml/classification/iris_classification.py

# نمونه اجرای خوشه‌بندی ارقام:
python 02-unsupervised-learning/kmeans_mnist.py
```

---

## 🛠️ کتابخانه‌ها و ابزارها (Technologies)
- **Python 3.10+**
- **Scikit-Learn:** KNN, LinearRegression, SGDClassifier, KMeans, StandardScaler
- **Pandas & NumPy:** Data preprocessing, matrix transformations
- **TensorFlow / Keras:** Multilayer Perceptron (MLP)
- **OpenCV & MTCNN:** Computer vision and face detection
- **Matplotlib:** Visualization
