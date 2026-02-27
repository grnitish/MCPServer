# """"
# MCP Server for News RAG System
# Exposes news search tool for Claude Desktop
# """

from mcp.server.fastmcp  import FastMCP
# from mcp.server.stdio import stdio_server
# from mcp.types import Tool,TextContext
from typing import List,Dict,Optional
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import requests
import trafilatura
import pandas as pd
import pyodbc
import asyncio
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

mcp = FastMCP("news-rag-server")
SQL_SERVER = "blueserver.database.windows.net"
SQL_DATABASE = "BlueriverInital"
SQL_USER = "readonly_user"
SQL_PWD = os.environ.get("SQL_PASSWORD")
# print(SQL_PWD)

# Usually {ODBC Driver 17 for SQL Server} or {ODBC Driver 18 for SQL Server}
SQL_DRIVER = "{ODBC Driver 18 for SQL Server}"

def get_db_connection():
    conn_str = (
        f"DRIVER={SQL_DRIVER};"
        f"SERVER={SQL_SERVER};"
        f"DATABASE={SQL_DATABASE};"
        f"UID={SQL_USER};"
        f"PWD={SQL_PWD};"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
        "Connection Timeout=30;"
    )
    return pyodbc.connect(conn_str)


@mcp.tool()
async def news(query:str ="Apple shares",num_results:int = 5)->List:
    """
    🌐 LIVE WEB SEARCH for real-time financial data and news.
    
    **USE THIS TOOL WHEN:**
    - User asks "What is the current price of [TICKER]?"
    - User asks "How is [STOCK] doing today?"
    - User wants LIVE/CURRENT market prices (e.g., "What's SLV trading at?")
    - User wants latest news about a company or stock
    - User wants current market information NOT in the portfolio database
    
    **DO NOT USE for:**
    - Historical portfolio transactions (use query_portfolio instead)
    - "How much did I spend on X?" (that's portfolio data)
    
    Args:
        query: Search query (e.g., "SLV current price", "RKLB stock news")
        num_results: Number of articles to fetch (default: 5)
    
    Returns:
        List of articles with source, content, and article_no
    
    Example queries:
    - "Tesla stock price today"
    - "SLV current price"
    - "Apple earnings news"
    """
    serper_api_key=os.getenv("SERPER_API")

    url = "https://google.serper.dev/search"

    payload = {
    "q": query
    }
    headers = {
    'X-API-KEY': serper_api_key ,
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, json=payload)

    response = response.json()

    # print(response )
    # print(response["topStories"])

    contents  = []
    try:
        for i in range(len(response["topStories"])):
            #print(new["link"])
            downloaded = trafilatura.fetch_url(response["topStories"][i]["link"])
            content = trafilatura.extract( downloaded, include_comments=False, include_tables=False)
            if content:
                contents.append({"article_no":i,"source":response["topStories"][i]["source"],"article":content})
    except Exception as e:
        # A general exception handler can be used as a fallback
        contents.append({"error":{str(e)}})

    """
    "answerBox": {
        "title": "Tesla Inc / Stock Price",
        "answer": "427.03 +1.82 (0.43%)",
        "source": "Disclaimer"
    },
    """
    try:
        if response["answerBox"]:

            contents.append({"title of Stock":response["answerBox"]["title"],"current price of stock":response["answerBox"]["answer"]})
    except Exception as e:
        # A general exception handler can be used as a fallback
        contents.append({"error":{str(e)}})
    print("I am here")
    return contents

@mcp.tool()
async def query_portfolio(sql_query: str) -> str:
    """
    📊 PORTFOLIO DATABASE - Query YOUR historical transaction records.

    - Amount (DECIMAL): Represents cash flow. 
        * NEGATIVE values (e.g., -72.53) mean money SPENT (a Buy).
        * POSITIVE values mean money RECEIVED (a Sell or Dividend).
    
    **USE THIS TOOL WHEN:**
    - User asks "How much did I spend on [STOCK]?"
    - User asks "How many shares of [STOCK] do I own?"
    - User wants to see their transaction history
    - User asks about PAST purchases, sales, dividends
    
    **DO NOT USE for:**
    - Current/live market prices (use news tool instead)
    - "What is [STOCK] trading at?" (that's live data, use news)
    
    DATABASE SCHEMA (dbo.portfolio):
    - Activity Date (DATE): Trade execution date
    - Process Date (DATE): System logging date
    - Settle Date (DATE): Settlement date
    - Instrument (VARCHAR): Stock ticker (e.g., 'RKLB', 'SLV')
    - ["Description"] (VARCHAR): Full asset name
    - "Trans_Code" (VARCHAR): 'BUY', 'SELL', 'DIV', 'INT'
    - Quantity (DECIMAL): Number of shares
    - Price (DECIMAL): Execution price per share
    - Amount (DECIMAL): Total value (Price × Quantity + Fees)
    
    Example queries:
    - "How much did I spend on RKLB?" 
      → SELECT SUM(Amount) FROM dbo.portfolio WHERE Instrument='RKLB' AND Trans_Code='BUY'
    - "How many SLV shares do I own?"
      → SELECT SUM(Quantity) FROM dbo.portfolio WHERE Instrument='SLV' AND Trans_Code='BUY'
    
    Args:
        sql_query: Must be a SELECT statement (other operations blocked like UPDATE , INSERT, DELETE , TURNCATE)
    
    Returns:
        JSON array of results or error message
    """
    # ... rest of your code

    # Safety Check: Prevent non-SELECT queries
    if not sql_query.strip().lower().startswith("select"):
        return "Error: Only SELECT queries are permitted for this readonly user."

    try:
        conn = get_db_connection()
        # Use pandas for easy JSON-like formatting
        df = pd.read_sql(sql_query, conn)
        print("✅ SUCCESS! Your driver and credentials are correct.")
        conn.close()
        
        if df.empty:
            return "Query successful, but no data was found."
            
        return df.to_json(orient="records", indent=2)
    
    except Exception as e:
        return f"Database Error: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport = "streamable-http")

