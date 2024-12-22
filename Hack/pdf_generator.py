import openai
import os
from chromadb import Client
from chromadb.utils import embedding_functions
from docx import Document
from typing import List
from dotenv import load_dotenv
from itertools import chain  # Import chain from itertools
import chromadb
from fpdf import FPDF

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize ChromaDB client
chroma_client = chromadb.Client()

# OpenAI embedding function for ChromaDB
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"), 
    model_name="text-embedding-ada-002",
    dimensions=1536  # Specify the number of dimensions for the chosen model
)

# Function to initialize the ChromaDB collection
def initialize_chromadb_collection() -> chromadb.Collection:
    collection_name = "contract_clauses"
    return chroma_client.get_or_create_collection(name=collection_name, embedding_function=openai_ef)

# Function to extract text from a Word document
def extract_text_from_word(doc_path: str) -> List[str]:
    doc = Document(doc_path)
    all_text = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():  # Remove empty paragraphs
            all_text.append(paragraph.text.strip())
    return all_text

# Function to store clauses in ChromaDB
def store_clauses_in_chromadb(collection: chromadb.Collection, clauses: List[str]):
    existing_ids = collection.get()["ids"]  # Get existing IDs in the collection
    for i, clause in enumerate(clauses):
        clause = clause.strip()
        # Skip unnecessary disclaimers or non-essential text
        if "this is a basic lease agreement" in clause.lower():
            continue
        clause_id = f"clause-{i}"
        if clause_id not in existing_ids:  # Add only if the ID doesn't exist
            collection.add(documents=[clause], ids=[clause_id], metadatas=[{"index": i}])
        else:
            print(f"ID {clause_id} already exists. Skipping this clause.")

# Function to retrieve relevant clauses based on a prompt
def retrieve_clauses(collection: chromadb.Collection, prompt: str, top_k: int = 5) -> List[str]:
    results = collection.query(query_texts=[prompt], n_results=top_k)
    retrieved_clauses = list(chain.from_iterable(results["documents"]))

    # Filter out disclaimers or non-relevant text
    filtered_clauses = [clause for clause in retrieved_clauses if "this is a basic lease agreement" not in clause.lower()]

    return filtered_clauses


# Function to generate a contract using OpenAI GPT
def generate_contract(prompt: str, retrieved_clauses: List[str]) -> str:
    context = "\n".join(retrieved_clauses)
    full_prompt = f"""
    Based on the following context:
    {context}

    Generate a contract for: {prompt}.
    Do not include disclaimers, notes, or additional comments in the output.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt},
        ],
        max_tokens=1500,
        temperature=0.2
    )
    
    return response['choices'][0]['message']['content'].strip()

file_path = r'c:\Users\21650\Desktop\langchain-rag-tutorial-main\langchain-rag-tutorial-main\chroma'
# Function to process and store clauses in the database (one-time setup)
def process_and_store_clauses(files_dir: str, collection: chromadb.Collection):
    for file_name in os.listdir(files_dir):
        if file_name.endswith(".docx"):
            file_path = os.path.join(files_dir, file_name)
            print(f"Processing file: {file_path}")
            clauses = extract_text_from_word(file_path)
            store_clauses_in_chromadb(collection, clauses)

# Function to generate PDF from contract text
def generate_pdf(content: str, output_pdf_path: str):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add the text, handling multi-line content
    for line in content.split("\n"):
        pdf.multi_cell(0, 10, line)

    pdf.output(output_pdf_path)


def clean_generated_contract(contract: str) -> str:
    # Split the contract into lines
    lines = contract.split("\n")
    
    # Remove lines containing "NOTE" or similar text
    cleaned_lines = [
        line for line in lines
        if not line.strip().lower().startswith("note:")
    ]
    
    # Join the cleaned lines back into a single string
    return "\n".join(cleaned_lines)

