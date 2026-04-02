# ============================================================
# search_server.py
# STABLE MCP SERVER using Tavily API
# ============================================================

from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient

# Replace with your actual key from tavily.com
TAVILY_API_KEY = ""

mcp = FastMCP("search")
tavily = TavilyClient(api_key=TAVILY_API_KEY)

@mcp.tool()
def search_web(query: str) -> str:
    """Search the web for real-time information.
    Use this for factual questions, historical data, or general lookups."""
    try:
        # depth="basic" is faster and costs 1 credit
        response = tavily.search(query=query, search_depth="basic", max_results=3)
        results = response.get('results', [])
        
        if not results:
            return f"No results found for: '{query}'"
            
        return "\n\n".join([
            f"[{i+1}] {r['title']}\n    {r['content']}"
            for i, r in enumerate(results)
        ])
    except Exception as e:
        return f"Search error: {e}"

@mcp.tool()
def search_news(query: str) -> str:
    """Search for the latest news articles on a topic.
    Use this for recent events, announcements, or developments within the last month."""
    try:
        # topic="news" triggers Tavily's news-specific crawler
        response = tavily.search(query=query, topic="news", search_depth="basic", max_results=3)
        results = response.get('results', [])
        
        if not results:
            return f"No news found for: '{query}'"
            
        return "\n\n".join([
            f"[{i+1}] {r['title']}\n"
            f"    Date: {r.get('published_date', 'Recent')}\n"
            f"    Content: {r['content']}\n"
            f"    Source: {r.get('url', 'Unknown')}"
            for i, r in enumerate(results)
        ])
    except Exception as e:
        return f"News search error: {e}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
