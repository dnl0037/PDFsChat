from typing import List, Union
import io
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS

PDFFile = Union[str, io.BytesIO, io.BufferedReader, io.TextIOWrapper]


def get_pdf_text(pdf_docs: List[PDFFile]) -> str:
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_text: str) -> List[str]:
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(raw_text)


def get_vector_store(text_chunks: List[str]):
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store


def main():
    load_dotenv()
    st.set_page_config(page_title="Multiple PDF Chat", page_icon=":books:")
    st.header("Multiple PDF Chat :books:")
    st.text_input("Ask a question about your PDFs:")
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf raw text
                raw_text = get_pdf_text(pdf_docs)
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # create vector storr
                vector_store = get_vector_store(text_chunks)


if __name__ == "__main__":
    main()
