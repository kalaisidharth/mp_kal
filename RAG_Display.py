import matplotlib
matplotlib.use("Agg")  # IMPORTANT: avoids Tkinter issues

import networkx as nx
import matplotlib.pyplot as plt

# Create graph
G = nx.DiGraph()
G.add_node("Aspirin", type="drug")
G.add_node("Inflammation", type="disease")
G.add_edge("Aspirin", "Inflammation", relation="treats")

# Layout
pos = nx.spring_layout(G, seed=42)

# Create figure
plt.figure(figsize=(8, 6))

# Draw graph
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=3000,
    node_color="lightblue",
    font_size=15
)

# Draw edge labels
edge_labels = nx.get_edge_attributes(G, "relation")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Save instead of show
plt.savefig("rag_graph.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved graph to rag_graph.png")


