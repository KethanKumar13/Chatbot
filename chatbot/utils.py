import os
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from text_to_doc import get_doc_chunks
from web_crawler import get_data_from_website
from prompt import get_prompt
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM


def get_chroma_client():

    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(
        collection_name="website_data",
        embedding_function=embedding_function,
        persist_directory="data/chroma"
    )

def store_docs(url):
   
    text, metadata = get_data_from_website(url)
    docs = get_doc_chunks(text, metadata)
    vector_store = get_chroma_client()
    vector_store.add_documents(docs)
    vector_store.persist()

def make_chain():
    
    model = OllamaLLM(
        model= "llama3.1",  
        temperature=0.0,
        verbose=True
    )
    vector_store = get_chroma_client()
    prompt = get_prompt()

    retriever = vector_store.as_retriever(search_type="mmr", verbose=False)

    chain = ConversationalRetrievalChain.from_llm(
        model,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs=dict(prompt=prompt),
        verbose=True,
        rephrase_question=False,
    )
    return chain


def get_response(question, organization_name, organization_info, contact_info):
    
    chat_history = ""
    chain = make_chain()
    response = chain({
        "question": question,
        "chat_history": chat_history,
        "organization_name": organization_name,
        "contact_info": contact_info,
        "organization_info": organization_info
    })
    return response['answer']
