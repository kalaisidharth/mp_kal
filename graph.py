from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


HUGGINGFACE_REPO = "HuggingFaceH4/zephyr-7b-beta"  # or another model you prefer

llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO,
        task="text-generation",
        # huggingfacehub_api_token="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"  # your HF token
    )
)

def manager_node(state):
    task_input=state.get("Task", "")
    input=state.get("Input", "")
    prompt=f"""You are a task router. Based on the user request below, decide whether it is a:
    -translate
    -summarize
    -calculate
    Respond with only one word (translate, summarize, calculate).
     task: {task_input}
     """
    decision=llm.invoke(prompt).content.strip().lower()
    return {"agent": decision, "input": input}

def translate_node(state):
    text=state.get("input", "")
    prompt=f"Act like you are translator.only respond with the English translation of the following text: {text}"
    result=llm.invoke(prompt).content
    return {"result": result}

def summarize_node(state):  
    text=state.get("input", "")
    prompt=f"Summarize the following text in one sentence: {text}"
    result=llm.invoke(prompt).content
    return {"result": result}
def calculate_node(state):  
    expression=state.get("input", "")
    prompt=f"Calculate the result of the following expression: {expression}"
    result=llm.invoke(prompt).content
    return {"result": result}

def default_node(state):
    return {"result":"Sorry, I could not understand the task."}

def route_by_agent(state):
    return {
        "translate": "Translator",
        "summarize": "Summarizer",
        "calculate": "Calculator",
        "input": state.get("input", "")
    }.get(state.get("agent", ""), "Default")

# from langgraph.graph import StateGraph
# g = StateGraph(dict)
# g.add_node("Manager", manager_node) 
# g.add_node("Translator", translate_node) 
# g.add_node("Summarizer", summarize_node) 
# g.add_node("Calculator", calculate_node) 
# g.add_node("Default", default_node)

# g.set_entry_point("Manager") 
# g.add_conditional_edges("Manager", route_by_agent)

# g.set_finish_point("Translator")
# g.set_finish_point("Summarizer")
# g.set_finish_point("Calculator")
# g.set_finish_point("Default")
# graph= g. compile()




from langgraph.graph import StateGraph
g = StateGraph(dict)
g.add_node("Manager", manager_node)
g.add_node("Translator", translate_node)
g.add_node("Summarizer", summarize_node)
g.add_node("Calculator", calculate_node)
g.add_node("Default", default_node)
g.set_entry_point("Manager")
g.add_conditional_edges("Manager", route_by_agent)
g.set_finish_point("Translator")
g.set_finish_point("Summarizer")
g.set_finish_point("Calculator")
g.set_finish_point("Default")
graph= g. compile()

print(graph.invoke({
    "Task": "can you translate this?",
    "Input": "Bonjour tout le monde"
}))

print(graph.invoke({
    "Task": "please summarize the following",
    "Input":"LangGraph helps you build flexible multi-agent workflows in Python."
}))

respcal=graph.invoke({
    "Task": "What is 12 * 8 + 5?",
    "Input":"12 * 8 + 5"
})

print(respcal["result"])

print(graph.invoke({
    "Task": "can you dance?",
    "Input":"foo"
}))

from PIL import Image
import io

png_bytes = graph.get_graph().draw_mermaid_png()
img = Image.open(io.BytesIO(png_bytes))
img.show()






from PIL import Image
import io

png_bytes = graph.get_graph().draw_mermaid_png()
img = Image.open(io.BytesIO(png_bytes))
img.show()
img.save("graph1.png")
print("Graph visualization saved as graph1.png")
import streamlit as st
from PIL import Image
import io

# Assuming you already built your graph as in your code
# graph = g.compile()

st.title("LangGraph Workflow Visualization")

# Generate PNG bytes from the graph
png_bytes = graph.get_graph().draw_mermaid_png()

# Convert bytes to PIL image
img = Image.open(io.BytesIO(png_bytes))

# Display directly in Streamlit
st.image(img, caption="Workflow Graph", use_column_width=True)


img.save("graph1.png")
print("Graph visualization saved as graph1.png")