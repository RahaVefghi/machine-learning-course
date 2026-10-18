# Machine Learning & Deep Learning Course Exercises
### تمرین‌ها و پروژه‌های عملی دوره یادگیری ماشین و یادگیری عمیق

مجموعه‌ای کامل و ساختاریافته از کدها، پروژه‌ها و تمرین‌های پیاده‌سازی شده در حوزه‌های **یادگیری ماشین کلاسیک (Classical ML)**، **یادگیری بدون نظارت (Unsupervised Learning)**، **یادگیری عمیق (Deep Learning)** و **بینایی ماشین (Computer Vision)**.

---

## 📁 ساختار مخزن (Repository Structure)

```text
machine-learning-course/
│
├── requirements.txt                      # پیش‌نیازها و کتابخانه‌های مورد نیاز
├── .gitignore                             # فایل‌های نادیده گرفته‌شده توسط گیت
│
├── 01-classical-ml/                       # یادگیری ماشین با Scikit-Learn
│   ├── classification/                    # دسته‌بندی با KNN، رگرسیون لجستیک و SGD
│   │   ├── iris_classification.py        # طبقه‌بندی گل زنبق (Iris)
│   │   ├── diabetes_classification.py    # پیش‌بینی دیابت با پیش‌پردازش داده‌ها
│   │   └── data/                          # دیتاست‌های Iris و Diabetes
│   ├── regression/                        # رگرسیون خطی و SGD Regressor
│   │   ├── student_regression.py          # پیش‌بینی نمرات دانشجویان بر اساس ساعات مطالعه
│   │   ├── petrol_regression.py           # تخمین مصرف سوخت
│   │   └── data/                          # دیتاست‌های مصرف سوخت و نمرات
│   └── preprocessing/                     # پیش‌پردازش و انکودینگ داده‌ها
│       ├── house_price_encoding.py        # انکودینگ فیچرهای مسکن (OneHot / LabelBinarizer)
│       ├── sklearn_encoding.py            # کار با OrdinalEncoder و OneHotEncoder
│       └── data/                          # دیتای HousesInfo.txt
│
├── 02-unsupervised-learning/              # خوشه‌بندی و یادگیری بدون نظارت
│   ├── kmeans_basic.py                    # خوشه‌بندی پایه با K-Means روی داده‌های ترکیبی
│   ├── kmeans_image.py                    # فشرده‌سازی و مصورسازی مراکز خوشه‌ها در ارقام (Digits)
│   ├── kmeans_mnist.py                    # خوشه‌بندی تصاویر ارقام دست‌نویس MNIST
│   └── data/                              # دیتاست فشرده mnist.npz
│
├── 03-deep-learning/                      # شبکه‌های عصبی مصنوعی (ANN/MLP/CNN)
│   ├── mnist-benchmarks/                  # بنچمارک و مقایسه جامع الگوریتم‌ها روی MNIST
│   │   ├── models_on_mnist.py             # مقایسه Logistic Regression, KNN, KMeans, MLP و CNN
│   │   └── mnist_knn.ipynb                # نوت‌بوک جوپیتر تحلیل و بررسی الگوریتم‌ها
│   ├── house-price-mlp/                   # پیش‌بینی قیمت مسکن با شبکه عصبی چندلایه (MLP)
│   │   ├── house_price_mlp.py
│   │   └── data/
│   ├── fire-detection/                    # پروژه تشخیص آتش در تصاویر
│   │   ├── fire_classification_knn.py     # طبقه‌بندی با KNN
│   │   ├── fire_detection_nn.py           # آموزش با شبکه عصبی عمیق
│   │   ├── fire_detection_model.py        # ارزیابی مدل آموزش دیده روی نمونه‌های تستی
│   │   ├── use_models.py                  # استنتاج (Inference) مدل
│   │   └── test_samples/                  # تصاویر نمونه برای تست مدل
│   ├── gender-detection/                  # پروژه تشخیص جنسیت از روی تصویر چهره
│   │   ├── training.py                    # استخراج چهره با MTCNN و آموزش دسته‌بند
│   │   ├── gender_detector_model.py       # تشخیص بلادرنگ جنسیت روی تصاویر ورودی
│   │   └── test_model/                    # تصاویر تست آقایان و خانم‌ها
│   └── captcha-solver/                    # حل خودکار کدهای کپچا با بینایی ماشین و CNN
│       ├── digit_extraction.py            # بخش‌بندی و کانتوریابی ارقام در تصویر کپچا
│       ├── training_captcha.py            # شبکه عصبی کانوولوشنی (CNN) با Data Augmentation
│       └── CAptcha.jpg                    # تصویر نمونه کپچا
│
└── 04-computer-vision/                    # پردازش تصویر و بینایی ماشین (OpenCV)
    ├── opencv-basics/                     # مبانی کار با تصاویر در OpenCV
    │   ├── cv_basics.py                   # رسم اشکال، کانال‌های رنگی و خواندن تصاویر
    │   ├── opencv_ex2.ipynb               # نوت‌بوک تمرین‌های فیلترگذاری و تبدیل‌های OpenCV
    │   └── (تصاویر نمونه)
    └── face-detection/                    # تشخیص چهره و لندمارک‌ها
        ├── face_detection_mtcnn.py        # تشخیص چهره با کتابخانه MTCNN و رسم نقاط کلیدی
        └── maryam.jpg                     # تصویر تست
```

---

## 🛠️ پیش‌نیازها و نصب (Installation & Setup)

برای اجرای پروژه‌ها پیشنهاد می‌شود از یک محیط مجازی (Virtual Environment) پایتون استفاده کنید:

```bash
# کلون کردن مخزن
git clone https://github.com/<YOUR_USERNAME>/machine-learning-course.git
cd machine-learning-course

# ساخت و فعال‌سازی محیط مجازی
python -m venv venv

# ویندوز (PowerShell):
.\venv\Scripts\Activate.ps1

# لینوکس یا مک:
source venv/bin/activate

# نصب پکیج‌ها
pip install -r requirements.txt
```

---

## 🚀 نحوه اجرای پروژه‌ها (Usage Examples)

### ۱. دسته‌بندی دیتای گل زنبق (Iris Classification):
```bash
python 01-classical-ml/classification/iris_classification.py
```

### ۲. مقایسه جامع مدل‌ها روی دیتای MNIST:
این اسکریپت مدل‌های مختلف شامل رگرسیون لجستیک، رگرسیون خطی، KNN با کاهش بعد PCA، خوشه‌بندی KMeans، شبکه عصبی پرسپترون چندلایه (MLP) و شبکه عصبی کانوولوشنی (CNN) را با هم مقایسه می‌کند:
```bash
python 03-deep-learning/mnist-benchmarks/models_on_mnist.py
```

### ۳. تشخیص چهره با MTCNN:
```bash
python 04-computer-vision/face-detection/face_detection_mtcnn.py
```

### ۴. تفکیک ارقام کپچا با OpenCV:
```bash
python 03-deep-learning/captcha-solver/digit_extraction.py
```

---

## 💡 نکات طراحی (Design Highlights)
- **مسیردهی پویا و نسبی (Relative Pathing)**: تمام اسکریپت‌ها طوری اصلاح شده‌اند که مستقل از محل اجرا، مسیر فایل‌های دیتای خود را به درستی پیدا می‌کنند.
- **لودر هوشمند MNIST**: در صورت نبود فایل‌های سنگین CSV، دیتاست به صورت خودکار از طریق توابع Keras بارگذاری می‌شود تا ریپازیتوری سبک و قابل حمل باقی بماند.
- **حفظ حریم خصوصی و استانداردهای گیت**: ویدیوها و آرشیوهای سنگین دوره نادیده گرفته شده و مخزن با حجم بهینه (کمتر از ۲۰ مگابایت) آماده انتشار است.
