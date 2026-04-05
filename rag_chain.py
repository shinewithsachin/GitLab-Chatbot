import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(api_key=None):
    """Builds and returns the LangChain RAG chain using LCEL."""
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API key is not set.")
        
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if not os.path.exists("faiss_index"):
        raise FileNotFoundError("FAISS vector store not found. Run data_loader.py first.")
        
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=api_key 
    )

    system_prompt = (
        "You are the GitLab Handbook Assistant — an expert AI designed to help "
        "employees and candidates understand GitLab's Handbook and Direction pages.\n\n"
        "RULES:\n"
        "1. ONLY answer questions related to GitLab's handbook, culture, values, "
        "product direction, engineering, people operations, finance, sales, marketing, "
        "legal, and security.\n"
        "2. If the question is unrelated to GitLab, politely decline and remind the user "
        "this assistant is focused on GitLab topics.\n"
        "3. If you don't know the answer from the provided context, say so honestly and "
        "encourage them to visit [handbook.gitlab.com](https://handbook.gitlab.com/) for the latest information.\n"
        "4. Be professional, clear, and encouraging. Use markdown for better formatting.\n"
        "5. When citing information, mention which section of the handbook it comes from "
        "when possible.\n"
        "6. CRITICAL: Whenever you mention 'handbook.gitlab.com', YOU MUST format it as a clickable markdown hyperlink exactly like this: [handbook.gitlab.com](https://handbook.gitlab.com/).\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # LCEL (LangChain Expression Language) pipeline bypassing the buggy langchain.chains installation
    rag_chain = (
        RunnablePassthrough.assign(
            context=(lambda x: x["input"]) | retriever | format_docs
        )
        | prompt
        | llm
        | StrOutputParser()
        | (lambda output: {"answer": output}) # Format dict to match frontend expectation
    )

    return rag_chain
