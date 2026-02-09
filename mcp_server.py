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



load_dotenv()

mcp = FastMCP("news-rag-server")


@mcp.tool()
async def news(query:str ="Apple shares",num_results:int = 5)->List:
    """
    This function gets the latest news based on the user request
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

    #print(response["topStories"]  )

    """
    Example :
     "topStories": [
    {
      "title": "2.5 Billion Reasons Investors Should Be Bullish on This Trillion-Dollar Stock, and 1 Reason to Be Fearful",
      "link": "https://www.theglobeandmail.com/investing/markets/stocks/AAPL-Q/pressreleases/98122/2-5-billion-reasons-investors-should-be-bullish-on-this-trillion-dollar-stock-and-1-reason-to-be-fearful/",
      "source": "The Globe and Mail",
      "date": "33 minutes ago",
      "imageUrl": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRWBm6qy_s4GLFOM8L0KCaSjNmY8QguJ-m_u5E1FXwxaWh4ASKWtAb6U-mT3g&s"
    }
    """
    contents  = []
    for i in range(len(response["topStories"])):
        #print(new["link"])
        downloaded = trafilatura.fetch_url(response["topStories"][i]["link"])
        content = trafilatura.extract( downloaded, include_comments=False, include_tables=False)
        if content:
          contents.append({"article_no":i,"source":response["topStories"][i]["source"],"article":content})



    return contents

if __name__ == "__main__":
    mcp.run(transport = "streamable-http")


# def news(query:str ="Apple shares",num_results:int = 5)->List:
#     """
#     This function gets the latest news based on the user request
#     """
#     serper_api_key=os.getenv("SERPER_API")

#     url = "https://google.serper.dev/search"

#     payload = {
#     "q": query
#     }
#     headers = {
#     'X-API-KEY': serper_api_key ,
#     'Content-Type': 'application/json'
#     }

#     response = requests.request("POST", url, headers=headers, json=payload)

#     response = response.json()

#     #print(response["topStories"]  )

#     """
#      "topStories": [
#     {
#       "title": "2.5 Billion Reasons Investors Should Be Bullish on This Trillion-Dollar Stock, and 1 Reason to Be Fearful",
#       "link": "https://www.theglobeandmail.com/investing/markets/stocks/AAPL-Q/pressreleases/98122/2-5-billion-reasons-investors-should-be-bullish-on-this-trillion-dollar-stock-and-1-reason-to-be-fearful/",
#       "source": "The Globe and Mail",
#       "date": "33 minutes ago",
#       "imageUrl": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRWBm6qy_s4GLFOM8L0KCaSjNmY8QguJ-m_u5E1FXwxaWh4ASKWtAb6U-mT3g&s"
#     }
#     """
#     contents  = []
    # for i in range(len(response["topStories"])):
    #     #print(new["link"])
    #     downloaded = trafilatura.fetch_url(response["topStories"][i]["link"])
    #     content = trafilatura.extract( downloaded, include_comments=False, include_tables=False)
    #     if content:
    #       contents.append({"article_no":i,"source":response["topStories"][i]["source"],"article":content})


#     return contents

 