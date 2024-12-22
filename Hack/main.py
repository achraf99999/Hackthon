from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from helpers.dto import SaveDocument,QueryRequest,QueryResponse
import os
from dotenv import load_dotenv
import openai
import chromadb
from chromadb.utils import embedding_functions
import chromadb.api.client
from langchain_chroma.vectorstores import Chroma
from chromadb.api.client import ClientAPI
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from helpers.helper import get_scraped_text
from rich import print
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import List, Optional
from docx import Document
from io import BytesIO

from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv
from helpers.get_content import get_content_from_bytes
import urllib

from helpers.prompts import text_prompt, image_prompt

from openai import OpenAI


# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# Check if the OpenAI API key is provided
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the .env file.")

# Set OpenAI API key for Langchain
openai.api_key = openai_api_key
openai_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  
)

model = ChatOpenAI(model="gpt-4o-mini")

from pdf_generator import main 

@app.post("/generate-contract/")
async def generate_contract_api(user_prompt: str):
    try:
        # Lancer la fonction principale pour générer le contrat
        main(user_prompt)
        
        # Définir le chemin du fichier généré
        pdf_path = "generated_contract33333333333.pdf"
        
        # Vérifier si le fichier a été généré
        if os.path.exists(pdf_path):
            return FileResponse(path=pdf_path, filename="contract.pdf", media_type="application/pdf")
        else:
            raise HTTPException(status_code=500, detail="Erreur dans la génération du fichier PDF.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# Initialize Chroma client
chroma_client = chromadb.HttpClient(host=DB_HOST, port=DB_PORT)
print("*************************************",DB_HOST,DB_PORT)
openai_embeddings =  embedding_functions.OpenAIEmbeddingFunction(
               model_name="text-embedding-3-large")

text_spliter=RecursiveCharacterTextSplitter(
    chunk_size=1000,chunk_overlap=200
)
   
def extract_text_from_word(doc_path: str) -> List[str]:
    doc = Document(doc_path)
    all_text = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():  # Remove empty paragraphs
            all_text.append(paragraph.text.strip())
    return all_text

import os
from pathlib import Path
import urllib.parse

def savedocument(input_doc):
    col_metadata = {
        "url": input_doc.url,   
        "category_name": input_doc.category_name,
        "language": input_doc.language,
    }
    
    # Check if the input is a local file
    if os.path.exists(input_doc.url):  # Check if it's a valid local file path
        file_path = Path(input_doc.url)
        if file_path.suffix == '.docx':
            print("Processing local .docx file:", input_doc.url)
            
            # Extract text from the Word document
            doc_text = "\n".join(extract_text_from_word(file_path))
            
            # Save to the database
            db_collection = chroma_client.get_or_create_collection(
                name="legal_documents_template",
                embedding_function=openai_embeddings
            )
            db_collection.add(
                documents=[doc_text],
                metadatas=[col_metadata],
                ids=[str(input_doc.url)],
            )
            print("Document saved successfully.")
        else:
            print("Unsupported file format:", file_path.suffix)
    else:
        # Assume it's a URL and process accordingly
        parsed_url = urllib.parse.urlparse(input_doc.url)
        if parsed_url.scheme in ['http', 'https']:
            print("Processing remote URL:", input_doc.url)
            
            text = get_scraped_text(input_doc.url)
            
            db_collection = chroma_client.get_or_create_collection(
                name="legal_documents",
                embedding_function=openai_embeddings
            )
            
            docs = text_spliter.split_text(text)
            metadatas = [col_metadata for _ in range(len(docs))]
            ids = [str(i) + str(input_doc.category_name) for i in range(len(docs))]
            
            print("Splitting and saving documents to the database.")
            db_collection.add(
                documents=docs,
                metadatas=metadatas,
                ids=ids,
            )
            print("Documents saved successfully.")
        else:
            print("Invalid URL or unsupported scheme:", input_doc.url)


def retrieve_docs(request: QueryRequest):
    # Retrieve the query and number of top results (k) from the request
    query = request.query
    k = request.k

    print("*******************************************",query)

    # Perform the query to Chroma DB
    db_collection = chroma_client.get_or_create_collection(name="legal_documents")
    print("*******************************************",query)
    results = db_collection.query(
        query_embeddings=openai_embeddings([query]),  # Assuming you have a function to encode your query
        n_results=k
    )
    
    # Extract the top-k results from the query
    ids = results['ids']
    scores = results['distances']  # Assuming distances are the similarity scores
    documents = results['documents']  # Assuming the original documents are stored in 'documents'
    print("************************",'\n'.join(documents[0]))
    return '\n'.join(documents[0])


def get_model_response(query,retrived_text):
    input_prompt = """
    You are a virtual assistant tasked with answering questions. You will receive two pieces of information:
    1. A question
    2. A retrieved document

    Your task is to provide a response based on the content of the retrieved document.

    # Question:
    {question}

    # Retrieved Document:
    {docs}
    """

    prompt = PromptTemplate(
    template=input_prompt,
    input_variables=["question", "docs"]
    )
    chain = prompt | model
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    result = chain.invoke( input = { "question": query, "docs" : retrived_text} )
    return result.content


###############################"
# 
# 
#





# API endpoint to process and store clauses
@app.post("/lawyer/store_clauses")
async def store_clauses(file: UploadFile = File(...)):
    file_bytes = await file.read()
    try:
        doc = BytesIO(file_bytes)
        clauses = extract_text_from_word(doc)
        collection = chroma_client.get_or_create_collection(
            name="legal_documents_template", embedding_function=openai_embeddings
        )
        store_clauses_in_chromadb(collection, clauses)
        return {"message": "Clauses stored successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# API endpoint to generate contract
@app.post("/lawyer/generate_contract")
async def create_contract(prompt: str):
    try:
        collection = chroma_client.get_or_create_collection(name="legal_documents_template")
        retrieved_clauses = retrieve_clauses(collection, prompt)
        contract = generate_contract(prompt, retrieved_clauses)
        cleaned_contract = "\n".join([line for line in contract.split("\n") if not line.lower().startswith("note:")])
        pdf_path = f"generated_contract_{prompt}.pdf"
        generate_pdf(cleaned_contract, pdf_path)
        return {"message": "Contract generated successfully.", "pdf_path": pdf_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# API endpoint to save a document
@app.post("/lawyer/save")
async def add_document(input_doc: SaveDocument, background_tasks: BackgroundTasks):
    try:
       
        # Schedule the document saving in the background with the client passed as an argument
        background_tasks.add_task(savedocument,input_doc)
        
        print("document saved successfully")
        
        return {"message": f"Document titled '{input_doc.url}' is being processed and will be saved to collection '{input_doc.category_name}' shortly."}
       
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")






@app.post("/lawyer/model_reponse")
async def retrieve_top_k(request: QueryRequest):
    try:
        print("**********************************************************test")
        # Retrieve documents synchronously instead of background tasks
        retrived_text  = retrieve_docs(request)
        print("*********************************************************************** hello ")

        model_response = get_model_response(request.query,retrived_text)
        # Return the retrieved documents along with their IDs and scores
        print("**********************************************************",model_response)
        return {
            "model_response": model_response
           
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/check_fraud")
async def check_fraud(file: UploadFile = File(...)):
    file_bytes = await file.read()
    try:
        document = get_content_from_bytes(file.filename, file_bytes)
    except ValueError as e:
        return {"error": str(e)}
    print("*******************************************************************bbbbbbbbbbbbbbbgo")
    try:
        if file.filename.endswith('.pdf'):
            response = model.invoke(
                image_prompt(document)
            )
        else:
            response = model.invoke(text_prompt(document))
        print("*******************************************************************bbbbbbbbbbbbbbbgo")
    except Exception as ex:
        return {"error": str(ex)}
    
    return {"response": response.content}

    
