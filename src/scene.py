import open3d as o3d
import numpy as np
import os
import cv2

class SceneLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        if not os.path.exists(self.file_path):
            print(f"Error: File '{self.file_path}' not found.")
            return None
        # Loads GLB model with full PBR material support
        print(f"Loading {self.file_path}...")
        return o3d.io.read_triangle_model(self.file_path)

