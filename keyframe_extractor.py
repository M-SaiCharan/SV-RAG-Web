import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score

class SemanticKeyframeExtractor:
    def __init__(self, model_id="openai/clip-vit-base-patch32"):
        # Detect Mac MPS or CPU
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"Loading CLIP on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def extract_frames(self, video_path, sample_rate=1):
        """Extracts frames at a fixed rate."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        timestamps = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * sample_rate) if sample_rate < 1 else int(fps / sample_rate)
        
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            count += 1
        cap.release()
        return frames, timestamps

    def get_embeddings(self, frames, batch_size=32):
        """Generates CLIP embeddings."""
        embeddings = []
        # Process in smaller batches to avoid RAM issues
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                embeds = self.model.get_image_features(**inputs)
                embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
                embeddings.append(embeds.cpu().numpy())
        return np.vstack(embeddings)

    def find_optimal_k(self, embeddings, min_k=5, max_k=25):
        """
        Research Novelty: Automatically finds the best K using Silhouette Score.
        """
        best_k = min_k
        best_score = -1
        
        # Ensure we don't look for more clusters than we have frames
        max_k = min(max_k, len(embeddings) - 1)
        if max_k <= min_k:
            return min_k

        print(f"Searching for optimal clusters between {min_k} and {max_k}...")
        
        for k in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
                
        print(f"Optimal K found: {best_k} (Silhouette Score: {best_score:.4f})")
        return best_k

    def cluster_and_select(self, frames, timestamps, n_clusters=None, use_adaptive=False):
        if not frames:
            return [], []
        
        embeddings = self.get_embeddings(frames)
        
        # LOGIC: If Adaptive is ON, we ignore the slider and calculate K
        if use_adaptive:
            n_clusters = self.find_optimal_k(embeddings)
        else:
            # Safety check
            n_clusters = min(n_clusters, len(frames))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        
        closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)
        selected_indices = sorted(closest_indices)
        
        keyframes = [frames[i] for i in selected_indices]
        key_timestamps = [timestamps[i] for i in selected_indices]
        
        return keyframes, key_timestamps