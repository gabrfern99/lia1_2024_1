import os
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
import pickle

class VisualSearchEngine:
    def __init__(self, image_directory, feature_file=None):
        self.image_directory = image_directory
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.features = []
        self.filenames = []
        self.nn = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if feature_file:
            self.load_features(feature_file)

    def extract_features(self, img):
        img = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(img)
        return features.cpu().numpy().flatten()

    def index_images(self):
        for filename in os.listdir(self.image_directory):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                filepath = os.path.join(self.image_directory, filename)
                img = Image.open(filepath).convert("RGB")
                features = self.extract_features(img)
                self.features.append(features)
                self.filenames.append(filename)
        
        self.features = np.array(self.features)
        self.nn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(self.features)

    def save_features(self, feature_file):
        with open(feature_file, 'wb') as f:
            pickle.dump((self.features, self.filenames), f)

    def load_features(self, feature_file):
        with open(feature_file, 'rb') as f:
            self.features, self.filenames = pickle.load(f)
        self.nn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(self.features)

    def search(self, query_image_path, num_results=5):
        query_img = Image.open(query_image_path).convert("RGB")
        query_features = self.extract_features(query_img)
        distances, indices = self.nn.kneighbors([query_features], n_neighbors=num_results)
        return [(self.filenames[i], distances[0][j]) for j, i in enumerate(indices[0])]

# Preprocess and save features
if __name__ == "__main__":
    image_directory = "/home/cr1sk/gitrepos/visual_se/images/"
    feature_file = "image_features.pkl"

    search_engine = VisualSearchEngine(image_directory)
    print("Indexing images...")
    search_engine.index_images()
    search_engine.save_features(feature_file)
    print("Indexing complete and features saved.")

