from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import OracleVS
from langchain_community.embeddings.oci_generative_ai import OCIGenAIEmbeddings
import os, oracledb

_ = load_dotenv(find_dotenv())
username = os.getenv("USERNAME")
password = os.getenv("PASSWORD")
dsn = os.getenv("DSN")
compartment_id = os.getenv("COMPARTMENT_ID")

def _initialize_vector_store():
    embedding_function = OCIGenAIEmbeddings(
        auth_type="INSTANCE_PRINCIPAL",
        service_endpoint="https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com",
        model_id="cohere.embed-multilingual-v3.0",
        compartment_id=compartment_id,
    )
    connection = oracledb.connect(
        user=username,
        password=password,
        dsn=dsn
    )
    vector_store = OracleVS(
        client=connection,
        embedding_function=embedding_function,
        table_name="sessions"
    )
    return vector_store

def main():
    loader = CSVLoader("./data/sessions.csv")
    documents = loader.load_and_split()
    vector_store = _initialize_vector_store()
    vector_store.add_documents(documents)

if __name__ == "__main__":
    main()
