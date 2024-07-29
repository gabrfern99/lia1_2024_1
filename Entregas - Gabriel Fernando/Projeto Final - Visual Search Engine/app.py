from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

class VisualSearchEngine:
    def __init__(self, feature_file):
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
        self.load_features(feature_file)

    def extract_features(self, img):
        img = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(img)
        return features.cpu().numpy().flatten()

    def load_features(self, feature_file):
        with open(feature_file, 'rb') as f:
            self.features, self.filenames = pickle.load(f)
        self.nn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(self.features)

    #def search(self, query_img, num_results=12):
    #    query_features = self.extract_features(query_img)
    #    distances, indices = self.nn.kneighbors([query_features], n_neighbors=num_results)
    #    return [(self.filenames[i], distances[0][j]) for j, i in enumerate(indices[0])]
    def search(self, query_image_path, num_results=12):
        query_features = self.extract_features(query_image_path)
        similarities = cosine_similarity([query_features], self.features)
        indices = similarities.argsort()[0][-num_results:][::-1]
        return [(self.filenames[i], similarities[0][i]) for i in indices]

# Initialize the search engine
feature_file = "image_features.pkl"
search_engine = VisualSearchEngine(feature_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    # Save the uploaded image to the static directory
    static_path = os.path.join('static', 'uploaded_image.jpg')
    file.save(static_path)
    
    img = Image.open(static_path).convert("RGB")

    # Get search results
    results = search_engine.search(img)

    # Prepare the paths for the input image and result images
    input_image_path = url_for('static', filename='uploaded_image.jpg')

    return render_template('results.html', 
                           results=results, 
                           input_image=input_image_path)

if __name__ == "__main__":
    app.run(port=10000, host="0.0.0.0", debug=True)

