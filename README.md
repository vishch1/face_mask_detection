# 😷 Face Mask Detection using Deep Learning

A real-time **Face Mask Detection System** built using **TensorFlow/Keras, MobileNetV2, and OpenCV**. The model classifies whether a person is wearing a face mask through a webcam feed and displays the prediction in real time.

---

## 📌 Features

* Real-time face mask detection using webcam
* Face detection using OpenCV Haar Cascade
* Mask classification using MobileNetV2
* Transfer learning for better accuracy
* Live prediction with bounding boxes and labels
* Training accuracy and loss visualization
* Lightweight and suitable for real-time applications

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* MobileNetV2
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn

---

## 📂 Project Structure

```
Face-Mask-Detection/
│
├── dataset/
│   ├── with_mask/
│   └── without_mask/
│
├── model/
│   ├── mask_detector.keras
│   └── training_plot.png
│
├── train_mask_model.py      # Train the model
├── detect_mask.py           # Real-time webcam detection
├── save_model.py            # Save trained model
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📥 Dataset

The dataset contains two classes:

* **with_mask**
* **without_mask**

Directory structure:

```
dataset/
│
├── with_mask/
│     image1.jpg
│     image2.jpg
│     ...
│
└── without_mask/
      image1.jpg
      image2.jpg
      ...
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Face-Mask-Detection.git

cd Face-Mask-Detection
```

---

### 2. Create a virtual environment (Optional)

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Train the Model

Run

```bash
python train_mask_model.py
```

The script will:

* Load the dataset
* Preprocess images
* Perform data augmentation
* Train MobileNetV2
* Evaluate the model
* Save the trained model
* Generate the training graph

Saved files:

```
model/
├── mask_detector.keras
└── training_plot.png
```

---

## 🎥 Run Real-Time Detection

```bash
python detect_mask.py
```

The webcam will start automatically.

Press

```
q
```

to exit.

---

## 🧠 Model Architecture

The project uses **Transfer Learning** with **MobileNetV2**.

Pipeline:

```
Input Image
      │
      ▼
MobileNetV2 (Pretrained on ImageNet)
      │
Average Pooling
      │
Flatten
      │
Dense (128, ReLU)
      │
Dropout (0.5)
      │
Dense (2 Classes, Softmax)
      │
Prediction
```

---

## 📈 Training

Hyperparameters:

| Parameter     | Value                           |
| ------------- | ------------------------------- |
| Image Size    | 224 × 224                       |
| Batch Size    | 8                               |
| Epochs        | 10                              |
| Learning Rate | 0.0001                          |
| Optimizer     | Adam                            |
| Loss Function | Sparse Categorical Crossentropy |

---

## 📊 Output

During real-time detection:

🟢 Green Bounding Box → **Mask**

🔴 Red Bounding Box → **No Mask**

Example:

```
-------------------------
|                       |
|      😀              |
|  [ Mask ]             |
|                       |
-------------------------
```

---

## 📈 Training Visualization

The training script automatically generates:

```
model/training_plot.png
```

The graph includes:

* Training Loss
* Validation Loss
* Training Accuracy
* Validation Accuracy

---

## 💡 Future Improvements

* Detect multiple faces simultaneously with improved accuracy
* Add face recognition along with mask detection
* Deploy using Flask or FastAPI
* Create a web application interface
* Improve detection under low-light conditions
* Convert the model to TensorFlow Lite for mobile deployment

---

## 📚 Learning Outcomes

This project demonstrates:

* Transfer Learning using MobileNetV2
* Image preprocessing
* Data augmentation
* CNN-based image classification
* Real-time computer vision with OpenCV
* Model evaluation using Classification Report
* Deep Learning model deployment

---

## 👩‍💻 Author

**Vishakha Chaudhary**

B.Tech Computer Science Engineering

**Skills**

* Java
* Python
* Spring Boot
* TensorFlow
* OpenCV
* React
* SQL
* AWS

---

## 📄 License

This project is developed for educational and learning purposes. Feel free to use and modify it for personal or academic projects.
