from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import os

import asyncio

class NewsAgentClient:
    load_dotenv()
    def __init__(self):
        self.client = MultiServerMCPClient(
        {
            "news-rag-server":{
                "url":"http://localhost:8000/mcp",
                "transport": "streamable_http"
            
            }
        }
    )
        self.model = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            temperature=0.3
        )
        self.agent = None

    async def initialize(self):
        """Fetches tools and sets up the ReAct agent."""
        if not self.agent:
            tools = await self.client.get_tools()
        
            self.agent = create_react_agent(self.model,tools)

    async def get_news(self,query:str):
        """Invokes the agent and returns the string content"""
        if not self.agent:
            await self.initialize()
        if self.agent is None:
            return "Error: Agent failed to initialize. Check your MCP server connection."
        response = await self.agent.ainvoke({"messages":[{"role":"user","content":query}]})

        return response["messages"][-1].content
