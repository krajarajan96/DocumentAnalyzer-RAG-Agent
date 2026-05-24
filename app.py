import os
import shutil
from typing import Annotated, Sequence, TypedDict
from operator import add as add_messages
import streamlit as st

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END

# --- CONFIGURATION & SETUP ---
load_dotenv()
st.set_page_config(page_title="Dynamic Document RAG Agent", layout="wide")

# Persistent directory for ChromaDB
DB_DIR = os.path.join(os.getcwd(), "chroma_db_storage")

# --- INITIALIZE STATE VARIABLES ---
if "rag_agent" not in st.session_state:
    st.session_state.rag_agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# --- CORE RAG & LANGGRAPH PROCESSING PIPELINE ---
def process_pdf_and_create_agent(uploaded_file):
    """Processes the uploaded PDF, builds an in-memory vector store, and compiles the agent."""
    # 1. Save uploaded file temporarily to disk for PyPDFLoader
    temp_pdf_path = f"temp_{uploaded_file.name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # 2. Extract and Split Text
    pdf_loader = PyPDFLoader(temp_pdf_path)
    pages = pdf_loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pages_split = text_splitter.split_documents(pages)
    
    # Clean up the temporary file immediately
    if os.path.exists(temp_pdf_path):
        os.remove(temp_pdf_path)

    # 3. Generate Embeddings & Build an In-Memory ChromaDB
    # By removing 'persist_directory', Chroma runs perfectly in RAM.
    # When a new file is uploaded, a fresh object is built and the old RAM space is garbage collected.
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        collection_name="dynamic_rag_collection"  # No persist_directory = Safe from file locks!
    )
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # 4. Define LangGraph Components
    @tool
    def retriever_tool(query: str) -> str:
        """Searches and returns relevant text chunks extracted from the uploaded document."""
        docs = retriever.invoke(query)
        if not docs:
            return "I found no relevant information in the uploaded document."
        return "\n\n".join([f"Document Segment {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

    tools = [retriever_tool]
    tools_dict = {t.name: t for t in tools}
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0).bind_tools(tools)

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def should_continue(state: AgentState):
        result = state['messages'][-1]
        return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

    system_prompt = f"""
    You are an intelligent AI assistant answering questions about the uploaded document: "{uploaded_file.name}".
    Use the retriever tool available to dig into the document data. You can make multiple calls if needed.
    Please always cite or reference specific contexts from the documents you use in your answers.
    """

    def call_llm(state: AgentState) -> AgentState:
        messages = [SystemMessage(content=system_prompt)] + list(state['messages'])
        return {'messages': [llm.invoke(messages)]}

    def take_action(state: AgentState) -> AgentState:
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            if t['name'] not in tools_dict:
                result = "Incorrect Tool Name. Select from available tools."
            else:
                with st.status(f"🔍 Running document search for: *'{t['args'].get('query')}'*..."):
                    result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        return {'messages': results}

    # 5. Build and Compile the Graph
    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever_agent", take_action)
    graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
    graph.add_edge("retriever_agent", "llm")
    graph.set_entry_point("llm")
    
    return graph.compile()


# --- UI LAYOUT ---
st.title("📄 Dynamic Document RAG Agent")
st.caption("Upload any PDF document to initialize your LangGraph RAG Agent instantly.")

# Sidebar for file uploading configuration
with st.sidebar:
    st.header("Document Control Center")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file:
        # Check if this is a newly uploaded file
        if st.session_state.current_file != uploaded_file.name:
            with st.spinner("Processing document and assembling LangGraph network..."):
                st.session_state.rag_agent = process_pdf_and_create_agent(uploaded_file)
                st.session_state.current_file = uploaded_file.name
                st.session_state.chat_history = []  # Flush chat history for new document context
            st.success(f"Agent ready for: {uploaded_file.name}")
    else:
        # Clear out current state if file is removed
        st.session_state.rag_agent = None
        st.session_state.current_file = None
        st.session_state.chat_history = []

# --- CHAT INTERFACE ---
if st.session_state.rag_agent:
    # Render historical chat logs safely
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        # FIX: Ensure it is an Assistant response AND has no tool_calls attached to it
        elif isinstance(msg, BaseMessage) and msg.content:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                continue # Skip the internal structural message that triggered the tool
            if isinstance(msg, ToolMessage):
                continue # Skip the raw text payload returned by the tool
                
            # If it passes those filters, it's the final written answer!
            with st.chat_message("assistant"):
                st.write(msg.content)

    # Listen for new user input
    if user_query := st.chat_input("Ask something about your document..."):
        # Display human message
        with st.chat_message("user"):
            st.write(user_query)
            
        # Append input to session state
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        
        # Invoke LangGraph through Streamlit UI pipeline
        with st.chat_message("assistant"):
            with st.spinner("Agent is analyzing..."):
                response = st.session_state.rag_agent.invoke({"messages": st.session_state.chat_history})
                
                # Retrieve final node answer message
                final_answer = response['messages'][-1].content
                st.write(final_answer)
                
                # Synchronize entire graph loop message state back to history tracker
                st.session_state.chat_history = response['messages']
else:
    st.info("👈 Please upload a PDF document in the sidebar to fire up the AI Agent.")