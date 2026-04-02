from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
import os, json, re, asyncio, subprocess, time, threading, requests
from langchain_mcp_adapters.client import MultiServerMCPClient
import nest_asyncio
nest_asyncio.apply()

mcp = MultiServerMCPClient({
    "math": {
        "command": sys.executable,     # Full Python path e.g. C:\anaconda3\python.exe
        "args": ["math_server.py"],    # Must be in same folder as notebook
        "transport": "stdio",          # Communicate via stdin/stdout pipes
    },
    "data": {
        "command": sys.executable,     # Full Python path e.g. C:\anaconda3\python.exe
        "args": ["data_server.py"],    # Must be in same folder as notebook
        "transport": "stdio",          # Communicate via stdin/stdout pipes
    },
    "search": {
        "command": sys.executable,     # Full Python path e.g. C:\anaconda3\python.exe
        "args": ["search_server.py"],    # Must be in same folder as notebook
        "transport": "stdio",          # Communicate via stdin/stdout pipes
    },
    # REMOTE server — already running, connect via HTTP
    "weather": {
        "url": "http://localhost:8000/mcp",
        "transport": "streamable_http",
    }
})


# ─── MCP Client Factory ───────────────────────────────────────────────────────
# Returns LangChain tools loaded from specified MCP servers

async def get_mcp_tools(servers: list) -> tuple:
    """
    servers: list of server names to connect to.
    Options: 'search', 'math', 'weather', 'data'
    Returns: (tools_list, tools_map, mcp_client)
    """
    tools = []
    print("Getting Tools")
    for server in servers:
        tool = await mcp.get_tools(server_name=server)
        tools.extend(tool)
    t_map  = {t.name: t for t in tools}
    print(f" MCP tools loaded: {list(t_map.keys())}")
    return tools, t_map

print(" MCP client helper ready")

tools, tools_map = await get_mcp_tools(["search", "math"])