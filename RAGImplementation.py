"""
Docstring for RAGImplementation
Now implementing RAG with news which will be fetched, processed, and stored.
"""

from typing import TypedDict, List, Dict , Annotated
from langgraph.graph import StateGraph, END
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

from pathlib import Path
import json
import os
from dotenv import load_dotenv
from typing import Any

load_dotenv()

class RAGState(TypedDict):
    question:str
    articles:List[Dict]
    documents: List[Document]
    retrieved_docs:List[Document]
    context:str
    answer:str
    sources:List[Dict]
    error:str
    vectorstore: Any


def load_news_node(state: RAGState)-> RAGState:
    print("\n📰 Node 1: Loading news articles...")
    all_articles = []
    try:
        folder = Path("./news_data")
        for file in folder.glob("*.json"):
            with open(file,'r',encoding='utf-8') as f:
                data =json.load(f)
            all_articles.extend(data.get('articles',[]))
            #articles = data.get('articles',[])

        state['articles'] = all_articles
        state['error'] = ""

        
    
    except Exception as e:
        state["error"] = f"Failed to load news data folder: {str(e)}"
        state['articles'] = []
    return state


def process_documents_node(state: RAGState) ->RAGState:
    print("\n📄 Node 2: Processing documents...")

    if state.get('error'):
        return state
    articles = state.get('articles')
    documents = []

    for i, article in enumerate(articles,1):

        content_parts = []

        if article.get('title'):
            content_parts.append(f"Title: {article['title']}")
        if article.get('snippet'):
            content_parts.append(f"Summary: {article['snippet']}")
        if article.get('source'):
            content_parts.append(f"Content: {article['source']}")
        if article.get('date'):
            content_parts.append(f"Date: {article['date']}")
        
        if article.get('content'):
            content_parts.append(f"\nFull Content:\n{article['content']}")
        content ="\n\n".join(content_parts)

        doc = Document(
            page_content = content,
            metadata ={
                "title": article.get('title',None),
                "source": article.get('source',None),
                "link": article.get('link',None),
                "date": article.get('date',None),
                "article_index":i

            }
        )
        documents.append(doc)

    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,

    )

    chunked_docs = text_spliter.split_documents(documents)
    state['documents'] = chunked_docs
    return state

def create_vectorstore_node(state:RAGState) ->RAGState:

    if state.get('error'):
        return state

    documents = state['documents']

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",model_kwargs={'device':'cpu'})

    vectorstore = Chroma.from_documents(
        documents,
        embedding = embeddings,
        persist_directory = './chroma_db_langgraph'    )

    state['vectorstore'] = vectorstore
    return state

def retrieve_documents_node(state:RAGState)->RAGState:

    if state.get('error'):
        return state
    
    question = state['question']
    vectorstore = state['vectorstore']

    if not vectorstore:
        state['error'] = "Vectorstore not initialized."
        return state
    
    retrieved_docs = vectorstore.similarity_search(
        question,
        k=5
    )

    for i,doc in enumerate(retrieved_docs,1):
        print(f"\n Retrieved Document {i}: {doc.metadata.get('title','No Title')}")

    state['retrieved_docs'] = retrieved_docs
    return state

def generate_answer_node(state:RAGState) ->RAGState:

    if state.get('error'):
        return state
    
    question = state['question']
    retrieved_docs = state['retrieved_docs']

    context_parts = []
    sources = []

    for i,doc in enumerate(retrieved_docs,1):
        context_parts.append(f"Article {i}:\n{doc.page_content}\n")
        sources.append({
            "title": doc.metadata.get('title','Unknown'),
            "link": doc.metadata.get('link',''),
            "source":doc.metadata.get('date','Unknown'),
            "date": doc.metadata.get('date','Unknown')   
        })

    context ="\n--\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system","""You are a helpful news assistant. Answer questions based ONLY on the provided context.
        
Rules:
1. Use ONLY information from the context
2. If the answer isn't in the context, say "I don't have enough information"
3. Be concise and accurate
4. Cite specific articles when possible"""
    ),
("user","""Context:{context}
 Question: {question}
 Answer: """)])
    
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        temperature=0.3
    )

    chain = prompt | llm

    response = chain.invoke(
        {"context" :context,"question":question}
        )
    answer =  response.content

    state['context'] = context
    state['answer'] = answer        
    state['sources'] = sources
    return state

# ============================================================================

def create_rag_graph():
    """
    Create LangGraph RAG workflow
    
    Returns:
        Compiled LangGraph workflow
    """
    # Create graph
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("load_news", load_news_node)
    workflow.add_node("process_documents", process_documents_node)
    workflow.add_node("create_vectorstore", create_vectorstore_node)
    workflow.add_node("retrieve_documents", retrieve_documents_node)
    workflow.add_node("generate_answer", generate_answer_node)
    
    # Define edges (workflow)
    workflow.add_edge("load_news", "process_documents")
    workflow.add_edge("process_documents", "create_vectorstore")
    workflow.add_edge("create_vectorstore", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    # Set entry point
    workflow.set_entry_point("load_news")
    
    # Compile
    app = workflow.compile()
    
    return app


class LangGraphRAG:
    """LangGraph-based RAG system"""
    
    def __init__(self):
        """Initialize RAG system"""
        print("🚀 Initializing LangGraph RAG System...")
        self.graph = create_rag_graph()
        self.vectorstore = None
        print("✅ RAG system ready!")
    
    def index_news(self, json_file: str = "news_data.json"):
        """
        Index news from JSON file
        
        Args:
            json_file: Path to JSON file
        """
        print(f"\n{'='*60}")
        print("📚 INDEXING NEWS ARTICLES")
        print('='*60)
        
        # Run indexing workflow (without question)
        initial_state = {
            "question": "",
            "articles": [],
            "documents": [],
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "sources": [],
            "error": ""
        }
        
        # Execute up to vectorstore creation
        result = self.graph.invoke(initial_state)
        
        # Store vectorstore for future queries
        self.vectorstore = result.get('vectorstore')
        
        print(f"\n{'='*60}")
        print("✅ INDEXING COMPLETE!")
        print('='*60)
    
    def ask(self, question: str) -> Dict:
        """
        Ask a question
        
        Args:
            question: User's question
            
        Returns:
            Answer with sources
        """
        print(f"\n{'='*60}")
        print(f"❓ QUESTION: {question}")
        print('='*60)
        
        if not self.vectorstore:
            return {
                "error": "Please index news first using index_news()"
            }
        
        # Create initial state
        initial_state = {
            "question": question,
            "articles": [],
            "documents": [],
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "sources": [],
            "error": "",
            "vectorstore": self.vectorstore
        }
        
        # Run only retrieval and generation
        # (Skip loading/processing since already done)
        from langchain_community.vectorstores import Chroma
        
        # Retrieve
        retrieved_docs = self.vectorstore.similarity_search(question, k=5)
        
        # Update state
        initial_state['retrieved_docs'] = retrieved_docs
        
        # Generate answer
        result = generate_answer_node(initial_state)
        
        print(f"\n{'='*60}")
        print("💡 ANSWER:")
        print('='*60)
        print(result['answer'])
        print(f"\n{'='*60}")
        print("📚 SOURCES:")
        print('='*60)
        
        for i, source in enumerate(result['sources'], 1):
            print(f"\n{i}. {source['title']}")
            print(f"   Source: {source['source']}")
            print(f"   Date: {source['date']}")
            print(f"   Link: {source['link']}")
        
        print('='*60)
        
        return {
            "answer": result['answer'],
            "sources": result['sources'],
            "context": result.get('context', '')
        }


# ============================================================================
# MAIN - DEMO USAGE
# ============================================================================

def main():
    """Main function"""
    
    print("\n" + "="*60)
    print("🎯 LANGGRAPH RAG SYSTEM FOR NEWS")
    print("="*60 + "\n")
    
    # Initialize RAG
    rag = LangGraphRAG()
    
    # Index news (one-time)
    rag.index_news("news_data.json")  # ← Your JSON file
    
    # Ask questions
    print("\n" + "="*60)
    print("💬 INTERACTIVE Q&A")
    print("="*60 + "\n")
    
    # Example questions
    example_questions = [
        "What did Tesla announce?",
        "What models is Tesla discontinuing?",
        "Tell me about Optimus robot",
        "What is Tesla's future strategy?"
    ]
    
    print("📝 Example questions:")
    for i, q in enumerate(example_questions, 1):
        print(f"{i}. {q}")
    
    print("\n" + "="*60)
    print("💡 Type your question (or 'quit' to exit)")
    print("="*60 + "\n")
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Get answer
            result = rag.ask(question)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


if __name__ == "__main__":
    main()