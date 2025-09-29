import logging
from typing import List, Dict, Any, Optional
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from database import db_manager
from config import settings

logger = logging.getLogger(__name__)

class RAGAgent:
    def __init__(self):
        self.llm = None
        self.memory = None
        self.agent = None
        self.retrieval_qa = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the RAG agent components"""
        try:
            # Initialize LLM
            self.llm = ChatOpenAI(
                model=settings.CHAT_MODEL,
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
                openai_api_key=settings.OPENAI_API_KEY
            )
            
            # Initialize memory
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                k=10  # Remember last 10 exchanges
            )
            
            # Initialize retrieval QA chain
            self._setup_retrieval_qa()
            
            # Initialize agent with tools
            self._setup_agent()
            
            logger.info("RAG agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG agent: {str(e)}")
            raise
    
    def _setup_retrieval_qa(self):
        """Setup the retrieval QA chain"""
        try:
            vector_store = db_manager.get_vector_store()
            
            # Custom prompt for RAG
            prompt_template = """Use the following pieces of context to answer the human's question. 
            If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
            Always cite the sources when providing an answer.

            Context: {context}

            Question: {question}

            Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            self.retrieval_qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_kwargs={"k": 5}  # Retrieve top 5 relevant documents
                ),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
        except Exception as e:
            logger.error(f"Failed to setup retrieval QA: {str(e)}")
            raise
    
    def _setup_agent(self):
        """Setup the agent with tools"""
        try:
            # Define tools
            tools = [
                Tool(
                    name="Document Search",
                    func=self._document_search_tool,
                    description="Search through uploaded documents to find relevant information. Use this when the user asks questions about their documents."
                ),
                Tool(
                    name="General Knowledge",
                    func=self._general_knowledge_tool,
                    description="Answer general knowledge questions that don't require document search."
                )
            ]
            
            # Initialize agent
            self.agent = initialize_agent(
                tools,
                self.llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=self.memory,
                verbose=True,
                max_iterations=3,
                early_stopping_method="generate"
            )
            
        except Exception as e:
            logger.error(f"Failed to setup agent: {str(e)}")
            raise
    
    def _document_search_tool(self, query: str) -> str:
        """Tool for searching documents"""
        try:
            if not self.retrieval_qa:
                return "Document search is not available. Please upload documents first."
            
            result = self.retrieval_qa({"query": query})
            return result["result"]
            
        except Exception as e:
            logger.error(f"Error in document search: {str(e)}")
            return "I encountered an error while searching documents. Please try again."
    
    def _general_knowledge_tool(self, query: str) -> str:
        """Tool for general knowledge questions"""
        try:
            response = self.llm.predict(query)
            return response
        except Exception as e:
            logger.error(f"Error in general knowledge: {str(e)}")
            return "I encountered an error while processing your question. Please try again."
    
    async def chat(self, message: str) -> Dict[str, Any]:
        """Process a chat message and return response with sources"""
        try:
            # Check if we have documents in the vector store
            vector_store = db_manager.get_vector_store()
            
            # First, try to get relevant documents
            relevant_docs = []
            sources = []
            
            try:
                retriever = vector_store.as_retriever(search_kwargs={"k": 5})
                relevant_docs = retriever.get_relevant_documents(message)
                sources = [doc.metadata.get("source", "Unknown") for doc in relevant_docs]
                sources = list(set(sources))  # Remove duplicates
            except Exception as e:
                logger.warning(f"Could not retrieve documents: {str(e)}")
            
            # If we have relevant documents, use retrieval QA
            if relevant_docs and self.retrieval_qa:
                try:
                    result = self.retrieval_qa({"query": message})
                    response = result["result"]
                    
                    # Get sources from the result
                    if "source_documents" in result:
                        source_docs = result["source_documents"]
                        sources = [doc.metadata.get("source", "Unknown") for doc in source_docs]
                        sources = list(set(sources))
                    
                except Exception as e:
                    logger.error(f"Error in retrieval QA: {str(e)}")
                    # Fallback to agent
                    response = self.agent.run(message)
            else:
                # Use agent for general conversation
                response = self.agent.run(message)
            
            return {
                "response": response,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return {
                "response": "I encountered an error while processing your message. Please try again.",
                "sources": []
            }
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector store"""
        try:
            vector_store = db_manager.get_vector_store()
            vector_store.add_documents(documents)
            
            # Refresh retrieval QA with new documents
            self._setup_retrieval_qa()
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            return False

# Global RAG agent instance
rag_agent = RAGAgent()