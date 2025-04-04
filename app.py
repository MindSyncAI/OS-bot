from flask import Flask, render_template, request, jsonify
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import pickle
import re

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize conversation chain
def initialize_chain():
    # Get API key
    groq_api_key = os.environ.get("GROQ_API_KEY")
    
    # Load pre-computed embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Load the vector store from disk with deserialization allowed
    vector_store = FAISS.load_local(
        "embeddings", 
        embeddings,
        allow_dangerous_deserialization=True  # We trust our own embeddings
    )
    
    # Load metadata
    with open("embeddings/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    print(f"Loaded {metadata['num_documents']} documents with {metadata['num_chunks']} chunks")
    
    # Initialize LLM
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama3-70b-8192",
        temperature=0.2,
        max_tokens=4000
    )
    
    # Initialize memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create custom prompt template
    custom_prompt = PromptTemplate(
        template="""You are an expert in Operating Systems. Use the following pieces of context to answer the question at the end. 
        If you don't find the exact answer in the context, use your general knowledge about operating systems to provide a comprehensive answer.
        Always provide detailed, accurate, and well-structured responses.
        
        Context: {context}
        
        Question: {question}
        
        Answer:""",
        input_variables=["context", "question"]
    )
    
    # Create conversation chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )
    
    return chain

# Initialize the chain at startup
conversation_chain = initialize_chain()

# Store conversation history
conversation_history = []

def format_response(text):
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    
    # Format each paragraph
    formatted_paragraphs = []
    for para in paragraphs:
        # Remove leading/trailing whitespace
        para = para.strip()
        
        # Skip empty paragraphs
        if not para:
            continue
            
        # Format lists
        if para.startswith(('- ', '* ', '1. ', '2. ', '3. ', '4. ', '5. ')):
            # Convert markdown lists to HTML-like structure
            para = para.replace('- ', '• ').replace('* ', '• ')
            para = re.sub(r'(\d+)\. ', r'\1. ', para)
        
        formatted_paragraphs.append(para)
    
    # Join paragraphs with proper spacing
    return '\n\n'.join(formatted_paragraphs)

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
    
    # Format the response
    formatted_answer = format_response(answer)
    
    # Update conversation history
    conversation_history.append((user_message, formatted_answer))
    
    return jsonify({'answer': formatted_answer})

@app.route('/reset', methods=['POST'])
def reset():
    global conversation_history
    conversation_history = []
    return jsonify({'status': 'success', 'message': 'Conversation reset successfully'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
