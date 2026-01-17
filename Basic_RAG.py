from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
import faiss
import os, getpass

os.environ['HUGGINGFACE_TOKEN']=getpass.getpass("Enter your Hugging Face token: ")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


tokenizer.pad_token = tokenizer.eos_token

def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    # print(inputs.keys())
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    embeddings = outputs.hidden_states[-1][:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)

    return embeddings

documents = [
    "A Christmas Carol is a novella by Charles Dickens, first published in London on 19 December 1843.",
    "The story tells of sour and stingy Ebenezer Scrooge's ideological, ethical, and emotional transformation after \n"
    "the supernatural visits of Jacob Marley and the Ghosts of Christmas Past, Present, and Yet to Come.",
    "The novella met with instant success and critical acclaim. It is regarded as one of the greatest Christmas stories ever written."
]
query = "What is the significance of Christmas Eve in A Christmas Carol?"               

query_embedding = get_embeddings([query])
document_embeddings = get_embeddings(documents)

print(document_embeddings.shape)
print(query_embedding.shape)

def cosine_similarity(embedding1, embedding2):
    return torch.nn.functional.cosine_similarity(embedding1, embedding2)

for doc_embedding in document_embeddings:
  print(cosine_similarity(query_embedding, doc_embedding))

similarities = [cosine_similarity(query_embedding, doc_embedding) for doc_embedding in document_embeddings]
print("Similarities:", similarities)

# documents
ranked_documents = sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)

top_documents = [doc for doc, _ in ranked_documents[:2]]
print(top_documents)

# Generation
augmented_input = query + " [SEP] " + " ".join(top_documents)
input_ids = tokenizer.encode(augmented_input, return_tensors="pt", padding=True, truncation=True)
print(input_ids)
outputs = model.generate(input_ids, max_length=150, num_beams=2, early_stopping=True)
print(outputs)

generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_response)

# Create FAISS index
index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(document_embeddings.numpy())
print("Number of documents in FAISS index:", index.ntotal)

# Retrieve information
query_embedding = get_embeddings([query])
distances, indices = index.search(query_embedding.detach().numpy(), k=5)
print("Distances:", distances[0])
print("Indices:", indices[0])

# Get top documents
top_documents = [documents[i] for i in indices[0]]
print("Top documents:", top_documents)


augmented_input = query + " [SEP] " + " ".join(top_documents)
     

# Generate the response
input_ids = tokenizer.encode(augmented_input, return_tensors="pt", padding=True, truncation=True)
print("Input IDs:", input_ids)
outputs = model.generate(input_ids, max_length=160, num_beams=2, early_stopping=True)
generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Response:", generated_response)

# With Chunks of Data
emb = np.random.random((10, 128)).astype('float32')
print("Embeddings shape:", emb.shape)
index = faiss.IndexFlatL2(128)
index.add(emb)
# Query
quer = np.random.random((1, 128)).astype('float32')
# search
distances, indices = index.search(quer, k=13)
print("Distances:", distances)
print("Indices:", indices)

stdoc = "The story tells of sour and stingy Ebenezer Scrooge's ideological, ethical, and emotional transformation after the supernatural visits of Jacob Marley and the Ghosts of Christmas Past, Present, and Yet to Come."
w = stdoc.split()
for i in range(0, len(w), 5):
  print(' '.join(w[i:i+5]))

# Function to chunk text
def chunk_text(text, max_length=100):
    words = text.split()
    chunks = [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]
    return chunks


chunks = []
for doc in documents:
    chunks.extend(chunk_text(doc, max_length=21))  # Adjust max_length as needed

print("Text Chunks:", chunks)

