from flask import Flask, render_template, request, jsonify
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS # type: ignore
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize conversation chain
def initialize_chain():
    # Get API key
    groq_api_key = os.environ.get("GROQ_API_KEY")
    
    documents = []
    pdf_files = []
    
    # Find all PDF files in the dataset directory
    for root, _, files in os.walk('dataset'):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    # Load each PDF file individually
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())
            print(f"Successfully loaded {pdf_file}")
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
    
    if not documents:
        raise ValueError("No documents were successfully loaded")
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': "cpu"}
    )
    
    # Create vector store
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    
    # Initialize LLM
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="mixtral-8x7b-32768",
        temperature=0.2,
        max_tokens=4000
    )
    
    # Initialize memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create conversation chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )
    
    return chain

# Initialize the chain at startup
conversation_chain = initialize_chain()

# Store conversation history
conversation_history = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_message = request.json.get('message', '')
    
    if not user_message:
        return jsonify({'answer': 'Please ask a question'})
    
    # Get response from the chain
    result = conversation_chain({"question": user_message, "chat_history": conversation_history})
    answer = result["answer"]
    
    # Update conversation history
    conversation_history.append((user_message, answer))
    
    return jsonify({'answer': answer})

@app.route('/reset', methods=['POST'])
def reset():
    global conversation_history
    conversation_history = []
    return jsonify({'status': 'success', 'message': 'Conversation reset successfully'})

if __name__ == '__main__':
    app.run(debug=True)