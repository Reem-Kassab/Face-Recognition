import numpy as np
import cv2
from keras_facenet import FaceNet
import tensorflow as tf

class FaceNetLoader:
    def __init__(self, model_path=None):
        print(f"[INFO] Initializing FaceNet...")
        # دي بتبني الموديل وتجهز كل حاجة أوتوماتيك
        self.embedder = FaceNet()
        
        if model_path:
            try:
                print(f"[INFO] Loading custom weights from: {model_path}")
                self.embedder.model.load_weights(model_path)
            except Exception as e:
                print(f"[WARNING] Could not load your weights: {e}")
                print("[INFO] Using default pre-trained FaceNet weights (Recommended).")
        
        # FaceNet expects 160x160 input
        self.input_shape = (160, 160)
        print("[INFO] Model loaded successfully.")

    def get_embedding(self, face_image):
        # 1. مكتبة keras-facenet بتحب تاخد الصور كـ Batch (مجموعة)
        # ولازم الصورة تكون RGB
        if face_image is None:
            return None
            
        # Resize to 160x160
        face_image = cv2.resize(face_image, self.input_shape)
        
        # Expand dims to become (1, 160, 160, 3)
        face_image = np.expand_dims(face_image, axis=0)
        
        # 2. المكتبة بتعمل الـ Preprocessing والـ Prediction في خطوة واحدة
        # هي بترجع ليستة، إحنا عايزين أول واحد بس
        embeddings = self.embedder.embeddings(face_image)
        
        return embeddings[0] # Return the 1D vector (512-d)