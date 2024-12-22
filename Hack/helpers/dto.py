from pydantic import BaseModel
from typing import List, Optional

# Define the Pydantic model for the document structure
class SaveDocument(BaseModel):
    url: str
    category_name: str
    language: str

class QueryRequest(BaseModel):
    query: str  # The query string to search the collection
    k: int = 5  # Number of top results to return (default to 5)


# DTO for the response
class QueryResponse(BaseModel):
    ids: List[str]  # List of IDs of the retrieved documents
    scores: List[float]  # List of similarity scores
    documents: List[str]  # List of document contents
