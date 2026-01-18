# 👤 Real-Time Face Recognition using FaceNet

## 📝 Description
This project implements a robust **Face Recognition System** capable of identifying individuals in **Real-Time**. 

Instead of using traditional methods or older architectures like VGGFace2, this implementation leverages **FaceNet**, a state-of-the-art deep learning model, to generate high-dimensional embeddings for faces. The system detects faces from a live webcam feed and matches them against a known database with high accuracy.

## 🚀 Key Features
* **FaceNet Architecture:** Utilizes FaceNet for generating precise 128-D facial embeddings.
* **Real-Time Processing:** Optimized for live detection and recognition via webcam.
* **High Accuracy:** Outperforms standard models by using advanced metric learning techniques.
* **Dynamic Database:** Easily add new faces to the recognition database.

## 🛠️ Technologies Used
* **Model:** FaceNet
* **Computer Vision:** OpenCV
* **Language:** Python
* **Libraries:** TensorFlow/Keras (or PyTorch depending on your implementation), NumPy

## 💻 How it Works
1.  **Database Creation:** Images of known individuals are processed, and their FaceNet embeddings are stored.
2.  **Detection:** The webcam captures frames, and faces are located using a face detector (e.g., Haar Cascades or MTCNN).
3.  **Recognition:** The detected face is passed through FaceNet to generate an embedding.
4.  **Matching:** The live embedding is compared to the stored database (using Euclidean distance or Cosine similarity) to find the best match.

## 🏃‍♀️ How to Run
1.  Clone the repository.
2.  Install requirements:
    ```bash
    pip install opencv-python numpy
    # Add other libraries you used, e.g., deepface or keras-facenet
    ```
3.  Run the application:
    ```bash
    python realtime.ipynb
    ```
