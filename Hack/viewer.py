import os
import chromadb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from rich import print
import requests
from helpers.get_content import get_content_from_bytes
from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer

load_dotenv()

st.set_page_config(
    page_title="db viewer",
)

GB_Port = "2015"
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

if "dbhost" not in st.session_state:
    st.session_state.dbhost = DB_HOST

@st.cache_resource()
def connectdb() -> chromadb.ClientAPI:
    return chromadb.HttpClient(host=st.session_state.dbhost, port=DB_PORT)

def viewcol():
    st.session_state.dbhost = st.text_input(
        label="db host", value=DB_HOST
    )
    if st.button(label="overwrite", use_container_width=True):
        st.cache_resource.clear()
        st.toast(
            body=f"Updated to {st.session_state.dbhost}",
        )

    client = connectdb()

    st.header("Viewer")
    list_clos = client.list_collections()

    collection_name = "legal_documents"

    chosen_col = client.get_collection(collection_name)
    data = {k: v for k, v in chosen_col.get().items() if k != "included"}
    df_to_display = pd.DataFrame(data)

    st.dataframe(df_to_display, use_container_width=True, height=1000)

def send_request():
    st.title('Save Docs')

    url = st.text_input("URL", "https://www.justia.com/family/")
    category_name = st.text_input("Category Name", "FamilyLaw")
    language = st.text_input("Language", "en")

    payload = {
        "url": url,
        "category_name": category_name,
        "language": language
    }

    api_url = "http://localhost:" + GB_Port + "/lawyer/save"

    if st.button('Send Request'):
        try:
            response = requests.post(api_url, json=payload)

            if response.status_code == 200:
                st.success("Request successfully sent!")
                st.json(response.json())
            else:
                st.error(f"Failed to send request. Status code: {response.status_code}")
                st.text(response.text)
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")

def chat():
    API_URL = "http://127.0.0.1:" + GB_Port + "/lawyer/model_reponse"

    st.title("Chatbot - Retrieve Similar Queries")

    user_input = st.text_input("Ask a question:")

    if user_input:
        payload = {
            "query": user_input,
            "k": 5
        }

        try:
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                data = response.json().get("model_response", None)
                print(response.json()["model_response"])
                if data:
                    st.write(f"Response: {data}")
                else:
                    st.write("No response.")
            else:
                st.error(f"Error fetching results. Status code: {response.status_code}")
        except Exception as e:
            st.error(f"Error occurred: {e}")

def fraud_detection():
    st.title("Document Fraud Detection")

    API_ENDPOINT = "http://127.0.0.1:" + GB_Port + "/check_fraud"

    if 'pdf_ref' not in ss:
        ss.pdf_ref = None

    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'txt', 'doc', 'docx'], key='pdf')

    if uploaded_file:
        ss.pdf_ref = uploaded_file

    if ss.pdf_ref:
        file_bytes = ss.pdf_ref.getvalue()

        st.subheader("File Preview")
        if ss.pdf_ref.name.endswith('.pdf'):
            pdf_viewer(input=file_bytes, width=700)
        else:
            preview = get_content_from_bytes(ss.pdf_ref.name, file_bytes)
            st.text(preview)

        with st.spinner('Analyzing document for fraud...'):
            files = {'file': (ss.pdf_ref.name, file_bytes)}
            response = requests.post(API_ENDPOINT, files=files)

            if response.status_code == 200:
                data = response.json()
                st.subheader("Analysis Results")
                st.write(data["response"])
            else:
                st.error(f"Error: {response.json()}")

def generate_contract():
    st.title("Generate Contract PDF")

    API_URL = "http://127.0.0.1:" + GB_Port + "/generate-contract/"

    user_prompt = st.text_area("Describe the contract:", height=200)

    if st.button("Generate PDF"):
        if user_prompt.strip():
            with st.spinner("Generating contract..."):
                try:
                    response = requests.post(API_URL, json={"user_prompt": user_prompt})

                    if response.status_code == 200:
                        st.success("Contract generated successfully!")
                        st.download_button(
                            label="Download Contract",
                            data=response.content,
                            file_name="contract.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please provide a description for the contract.")

def main():
    st.title("Welcome to the Dashboard!")
    st.write("Hello, welcome to the dashboard. Please choose an action below:")

    choice = st.radio(
        "Choose an action",
        ("Send Request", "Viewcol", "Chat", "fraud_detection", "Generate Contract PDF")
    )

    if choice == "Send Request":
        send_request()

    elif choice == "Viewcol":
        viewcol()

    elif choice == "Chat":
        chat()

    elif choice == "fraud_detection":
        fraud_detection()

    elif choice == "Generate Contract PDF":
        generate_contract()

if __name__ == "__main__":
    main()
