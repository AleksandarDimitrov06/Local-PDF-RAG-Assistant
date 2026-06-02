import streamlit as st
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

@st.cache_resource
def load_llm_and_embeddings():
    print("Loading Local LLM (this will take a while the first time to download)...")

    model_id = "Qwen/Qwen2.5-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        clean_up_tokenization_spaces=False
    )

    quant_config = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.5,
        do_sample=False,
        repetition_penalty=1.2,
        return_full_text=False
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return llm, embeddings, tokenizer


llm, embeddings, tokenizer = load_llm_and_embeddings()


def mainFunc(pdf_file):
    print("Loading PDF document...")

    loader = PyPDFLoader(pdf_file)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    print("Building embeddings...")
    vector_store = InMemoryVectorStore.from_documents(all_splits, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    
    def build_qwen_prompt(inputs: dict) -> str:
        context = inputs["context"]
        question = inputs["question"]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the question using ONLY the context below. "
                    "Be concise. Do not repeat yourself. "
                    "If the answer is not in the context, say you do not have that information.\n\n"
                    f"Context:\n{context}"
                )
            },
            {
                "role": "user",
                "content": question
            }
        ]

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | RunnableLambda(build_qwen_prompt)
        | llm
    )

    return rag_chain


def AskQuestion(chain, question):
    response = chain.invoke(question)
    return response