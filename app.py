import tempfile
import os
import streamlit as st
from main import AskQuestion, mainFunc

st.title("AI PDF Assistant")


if "messages" not in st.session_state:
    st.session_state.messages = []

pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"], key="pdf_uploader")

if pdf_file is not None:
    
    if "rag_chain" not in st.session_state or st.session_state.get("pdf_name") != pdf_file.name:
        with st.spinner("Processing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(pdf_file.read())
                temp_path = temp_file.name

            st.session_state.rag_chain = mainFunc(temp_path)
            st.session_state.pdf_name = pdf_file.name  
            st.session_state.messages = []             

            os.unlink(temp_path)

        st.success(f"✅ '{pdf_file.name}' processed successfully!")

    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

   
    question = st.chat_input("Ask a question about the PDF...")

    if question:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Generate and show assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = AskQuestion(st.session_state.rag_chain, question)
            st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    
    if "rag_chain" in st.session_state:
        del st.session_state.rag_chain
        del st.session_state.pdf_name
        st.session_state.messages = []

    st.info("👆 Upload a PDF to get started.")

# 'streamlit run app.py' to run the app in terminal
# 'ctrl + c' to stop the app