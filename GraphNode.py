from typing import TypedDict, List, Tuple
from langgraph.graph import StateGraph

from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
import os,getpass
os.environ["HF_TOKEN"] = getpass.getpass("Enter your Hugging Face API Token: ")
# ---------------- LLM ----------------
hf_pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

llm = HuggingFacePipeline(pipeline=hf_pipe)

# ---------------- Knowledge Graph ----------------
class KnowledgeGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, node_id, **attrs):
        self.nodes[node_id] = attrs

    def add_edge(self, src, relation, dst):
        self.edges.setdefault(src, []).append((relation, dst))

    def neighbors(self, node_id):
        return self.edges.get(node_id, [])

kg = KnowledgeGraph()
kg.add_node("Aspirin", type="drug")
kg.add_node("COX-1", type="enzyme")
kg.add_node("prostaglandins", type="chemical")
kg.add_node("Inflammation", type="effect")

kg.add_edge("Aspirin", "inhibits", "COX-1")
kg.add_edge("COX-1", "produces", "prostaglandins")
kg.add_edge("prostaglandins", "causes", "Inflammation")

# ---------------- State ----------------
class GraphRAGState(TypedDict):
    query: str
    knowledge_graph: KnowledgeGraph
    entities: List[str]
    graph_facts: List[Tuple[str, str, str]]
    context: str
    answer: str

# ---------------- Nodes ----------------
def extract_entities(state: GraphRAGState) -> GraphRAGState:
    # naive entity extraction for demo
    entities = []
    for node in state["knowledge_graph"].nodes:
        if node.lower() in state["query"].lower():
            entities.append(node)
    state["entities"] = entities
    return state


def traverse_graph(state: GraphRAGState) -> GraphRAGState:
    kg = state["knowledge_graph"]
    facts = []

    def dfs(node, depth, visited):
        if depth > 3 or node in visited:
            return
        visited.add(node)
        for rel, nbr in kg.neighbors(node):
            facts.append((node, rel, nbr))
            dfs(nbr, depth + 1, visited)

    for e in state["entities"]:
        dfs(e, 0, set())

    state["graph_facts"] = facts
    return state


def build_context(state: GraphRAGState) -> GraphRAGState:
    state["context"] = "\n".join(
        f"{s} {r} {o}." for s, r, o in state["graph_facts"]
    )
    return state


def answer_llm(state: GraphRAGState) -> GraphRAGState:
    prompt = f"""
Explain step by step how Aspirin reduces inflammation.
You must mention ALL intermediate steps.
Use only the facts provided.

Facts:
{state["context"]}

Step-by-step explanation:
"""
    state["answer"] = llm.invoke(prompt).strip()
    return state

# ---------------- Graph ----------------
graph = StateGraph(GraphRAGState)

graph.add_node("extract_entities", extract_entities)
graph.add_node("traverse_graph", traverse_graph)
graph.add_node("build_context", build_context)
graph.add_node("answer", answer_llm)

graph.set_entry_point("extract_entities")
graph.add_edge("extract_entities", "traverse_graph")
graph.add_edge("traverse_graph", "build_context")
graph.add_edge("build_context", "answer")

app = graph.compile()

# ---------------- Run ----------------
initial_state: GraphRAGState = {
    "query": "How does Aspirin reduce inflammation?",
    "knowledge_graph": kg,
    "entities": [],
    "graph_facts": [],
    "context": "",
    "answer": ""
}

result = app.invoke(initial_state)

print("\n-- GRAPH CONTEXT --")
print(result["context"])

print("\n-- ANSWER --")
print(result["answer"])
