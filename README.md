# 👗 Fashion Recommendation System with MobileNetV2

## 📌 Overview
This project builds a **content-based fashion recommendation system** using **MobileNetV2** image embeddings.
Given a query product image, the system retrieves visually similar items and (optionally) filters/boosts by metadata (e.g., *articleType*, *gender*, *season*).


## 🎯 Problem Statement
E-commerce platforms need fast, high-quality product discovery. Traditional keyword search is brittle for visual similarity. 
This system leverages **deep visual embeddings** to enable *"find similar"* recommendations that improve conversion and user experience.


## 🧠 Approach
- Backbone: **MobileNetV2** as a feature extractor
- `include_top=False` to obtain compact feature maps
- Pretrained weights: **None**
- Embedding extraction for all catalog images
- Fast nearest-neighbor lookup for retrieval

## 🗂️ Dataset & Preprocessing
- Resize images and normalize to `[0,1]` or `[-1,1]` depending on preprocessing

## ⚙️ Dependencies
```bash
pip install tensorflow pillow opencv-python numpy pandas scikit-learn matplotlib
# Optional for large-scale search
pip install faiss-cpu  # or faiss-gpu
```


## 🚀 How to Run
1. **Prepare data**: place images under a root folder and (optionally) a CSV with metadata.
2. **Extract embeddings**:
   - Load MobileNetV2 with `include_top=False` and `weights=None`.
   - Preprocess each image and run a forward pass to get a fixed-length vector (e.g., with global average pooling).
3. **Build index**:
   - Use NearestNeighbors/FAISS to index all vectors.
4. **Query**:
   - For a query image, compute its embedding and retrieve top-`k` neighbors (default k=10).
   - Return product IDs/paths and distances (euclidean/cosine).
5. **(Optional) Re-rank**:
   - Re-weight by metadata similarity (e.g., same `articleType`) or business rules.


## 🏋️‍♂️ Training
- This notebook primarily **extracts embeddings** (no supervised training). Enable fine-tuning for domain adaptation if desired.

## 📊 Evaluation
- Compute **Precision@K**, **Recall@K**, and **nDCG** using articleType as relevance (same-type = relevant).
- (Optional) User study: click-through-rate on recommendations

## ✅ Results
- After embedding extraction and KNN/FAISS search, visual inspection shows coherent retrievals for color, texture, and silhouette.
- Add a table or image grid with a few **query → top-5** recommendations for illustration.

## 🖼️ Visualization
- Side-by-side plots of the query and top-K retrieved items are generated in the notebook.
<img width="1489" height="352" alt="image" src="https://github.com/user-attachments/assets/a3797418-6c83-428b-99bb-814fbe1d844d" />
<img width="1490" height="305" alt="image" src="https://github.com/user-attachments/assets/2d674b8d-62e8-480e-a544-235a64c835c4" />
<img width="1490" height="305" alt="image" src="https://github.com/user-attachments/assets/9e987250-83a5-4d20-9c99-308e9257eaee" />
<img width="1490" height="269" alt="image" src="https://github.com/user-attachments/assets/857fcc0c-dac7-48eb-81a5-4ad424b4fba8" />


## 📁 Repository Structure
```
├── fashion-recommendation-system-with-mobilenetv2.ipynb  # Main notebook
└── README.md                                             # This documentation
```


## 🚀 Production Tips
- Cache embeddings and serve via a vector DB (FAISS, Milvus, Pinecone).
- Batch updates for new items; re-index incrementally.
- Add business rules: inventory, price, diversity, category constraints.
- Expose as API (FastAPI) and a simple UI (Streamlit/Next.js).


## 👨‍💻 Author
Built as a **content-based visual recommender** using MobileNetV2 embeddings.
