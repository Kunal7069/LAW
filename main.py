import os
import re
import pickle
import numpy as np
from wtpsplit import SaT
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
port = 8888
folder_path = "./files" 

def initialize_model():
    model = SaT("sat-3l-sm")
    # Uncomment to use GPU if available
    # model.half().to("cuda")
    return model

# Step 2: Extract legal references from segmented sentences
def extract_legal_references(segmented_sentences):
    pattern = r'\b(IPC|Section(?:s)?|Act(?:s)?)\b \d+'
    return [sentence for sentence in segmented_sentences if re.search(pattern, sentence, re.IGNORECASE)]

# Step 3: Process input text to extract legal references
def process_input_text(model, input_text):
    segmented_sentences = model.split(input_text)
    legal_references = extract_legal_references(segmented_sentences)
    return legal_references

# Step 4: Load graph embeddings
def load_graph_embeddings(graph_pkl_path):
    with open(graph_pkl_path, "rb") as f:
        G = pickle.load(f)
    nodes = list(G.nodes)
    node_embeddings = {node: G.nodes[node].get('embedding') for node in nodes}
    print("Graph loaded successfully.")
    return node_embeddings

# Step 5: Compute cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_top_similar_cases(input_embedding, node_embeddings, top_k=10):
    similarities = {
        node: cosine_similarity(input_embedding, np.array(embedding))
        for node, embedding in node_embeddings.items() if embedding is not None
    }
    return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]

def start(input_text):
    print("Processing input text to extract legal references...")
    segmentation_model = initialize_model()
    legal_references = process_input_text(segmentation_model, input_text)
    print("Extracted Legal References:", legal_references)

    # Combine legal references into a single text block for embedding
    combined_references = " ".join(legal_references) if legal_references else input_text

    # Load pre-trained sentence transformer model
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    input_embedding = sentence_model.encode(combined_references, normalize_embeddings=True)

    # Load graph embeddings
    graph_pkl_path = "semantic_similarity_graph.pkl"  # Replace with your graph path
    node_embeddings = load_graph_embeddings(graph_pkl_path)

    # Find top similar cases
    top_k = 10
    top_similar_cases = find_top_similar_cases(input_embedding, node_embeddings, top_k=top_k)

    # Display results
    print("\nTop similar cases based on legal references:")
    result=[]
    for rank, (case, score) in enumerate(top_similar_cases, start=1):
        demo=[]
        file_path = os.path.join(folder_path, case)  # Full path to the file
        
        try:
            # Read the file content
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            demo.append(rank)
            demo.append(content)
            demo.append(float(score))
            result.append(demo)
        
        except FileNotFoundError:
            print(f"File '{case}' not found in {folder_path}.")
        except Exception as e:
            print(f"Error reading file '{case}': {e}")
    return result
# Step 6: Main execution flow

@app.route('/process', methods=['POST'])
def process_input():
    try:
        data = request.get_json()

        if 'input_text' not in data:
            return jsonify({"error": "input_text is required"}), 400

        input_text = data['input_text']

        # Process the input using the existing logic
        result = start(input_text)

        # Return the result as a JSON response
        return jsonify({"result": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
    
