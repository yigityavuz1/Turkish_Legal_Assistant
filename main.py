import os
import streamlit as st
import pandas as pd
from src.embed_documents import get_db_instance  
from src.rag_pipeline import rag

# The page setup
st.set_page_config(page_title="Türk Anayasa Asistanı", layout="wide")
st.title("Türk Anayasa Asistanı")

@st.cache_resource
def load_vector_store(pdf_file: str):
    """
    Creates or loads Weaviate vector stores from the local PDF.
    """
    page_db, chunk_db = get_db_instance(pdf_file)
    return page_db, chunk_db

# Example query list
query_list = [
    "Temel hak ve hürriyetlerin sınırlanması hangi durumlarda gerçekleşebilir?",
    "Cumhurbaşkanının görev ve yetkileri nelerdir?",
    "Kanunların Anayasaya uygunluğunu kim denetler?",
]

# PDF path (adjust to your environment)
pdf_file = "data/gerekceli_anayasa.pdf"
page_db, chunk_db = load_vector_store(pdf_file)

selected_question = st.selectbox("Önerilen Sorular", [""] + query_list)
user_query = st.text_input("Sorunuz:", value=selected_question)

if user_query:
    with st.spinner("Yanıt oluşturuluyor..."):
        response, docs_content, docs_metadata = rag(user_query, page_db, chunk_db)
        
        st.markdown("### Cevap")
        st.write(response)

        st.markdown("### Kaynak Dökümanlar")
        st.write(docs_metadata)

        st.markdown("### Kaynak İçeriği")
        st.write(docs_content)

        if st.button("Cevabı Excel'e Kaydet"):
            qa_df = pd.DataFrame({"Soru": [user_query], "Cevap": [response]})
            qa_df.to_excel("anayasa_qa.xlsx", index=False)
            st.success("Cevap Excel Olarak Kaydedildi.")
