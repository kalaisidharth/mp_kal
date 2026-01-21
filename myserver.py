import fastmcp import FastMCP

mcp=FastMCP("My MCP Server")

def greet(name:str)-> str:
    return f"Hello, {name}!"

if __name__=="__main__":
    mcp.run(host="localhost",port=8000)