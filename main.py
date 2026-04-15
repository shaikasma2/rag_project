from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import pipeline
import os

# 1. Load PDFs
documents = []

for file in os.listdir("data"):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(f"data/{file}")
        docs = loader.load()
        
        for d in docs:
            d.metadata["source"] = file
        
        documents.extend(docs)

print("✅ PDFs loaded")

# 2. Split text
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

print("✅ Chunks created:", len(chunks))

# 3. Embeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(
    chunks,
    embedding,
    persist_directory="vector_db"
)

print("✅ Stored in DB")

# 4. Model
qa_pipeline = pipeline("text-generation", model="distilgpt2")

# 5. Chat
while True:
    query = input("\n💬 Ask a question (type 'exit' to stop): ")

    if query.lower() == "exit":
        break

    results = db.similarity_search(query, k=3)

    context = "\n".join([r.page_content for r in results])

    prompt = f"""
    Answer based only on this:

    {context}

    Question: {query}
    """

    response = qa_pipeline(prompt, max_length=200, num_return_sequences=1)

    print("\n🤖 Answer:", response[0]["generated_text"])

    print("\n📌 Sources:")
    for r in results:
        print(f"{r.metadata['source']} - page {r.metadata.get('page', 'N/A')}")