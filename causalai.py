import pandas as pd
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
import graphviz
import networkx as nx
import matplotlib.pyplot as plt
##import community as community_louvain
import community.community_louvain as community_louvain

#from graphutils import causal_dag
#import pygraphviz as pgv

# Generate synthetic marketing data
np.random.seed(42)
data = pd.DataFrame({
    "TV_Ads": np.random.randint(100, 500, 100),
    "Radio_Ads": np.random.randint(50, 200, 100),
    "Social_Media": np.random.randint(20, 100, 100),
    "Marketing_Budget": np.random.randint(500, 2000, 100),
    "Economic_Conditions": np.random.normal(0, 1, 100),
    "Sales": np.random.randint(1000, 5000, 100)
})

# Display first few rows
print(data.head())


# Convert data to NumPy array
data_array = data.to_numpy()

# Apply PC algorithm for causal discovery
cg = pc(data_array, alpha=0.05)  # alpha = significance level
print(cg)

##cg.draw_pydot_graph()

# # Visualize learned DAG
# pyd = GraphUtils.to_pydot(cg.G)
# pyd.write_png(r'C: /Users/Radha/Desktop/simple_test.png')

cg.to_nx_graph()
cg.draw_nx_graph(skel=False)

# Extract edges from causal discovery graph
# edges = [(cg.G.nodes[i], cg.G.nodes[j]) for i, j in zip(*np.where(cg.G.graph > 0))]
#
# # Convert discovered causal structure to graphutils DAG
# dag = causal_dag(edges)
#
# # Plot DAG
# plt.figure(figsize=(8, 6))
# nx.draw(dag, with_labels=True, node_size=3000, node_color="lightblue", edge_color="black", font_size=10)
# plt.title("Discovered Causal DAG for Market Mix Modeling")
# plt.show()



# Convert causal graph to NetworkX format
causal_graph = nx.DiGraph()

# Extract edges from the adjacency matrix
nodes = list(data.columns)
for i in range(len(nodes)):
    for j in range(len(nodes)):
        if cg.G.graph[i, j] == 1:  # If an edge exists
            causal_graph.add_edge(nodes[i], nodes[j])

# Draw causal DAG
nx.draw(causal_graph, with_labels=True, node_color="lightblue", edge_color="black", font_size=10)


# Apply Louvain clustering
partition = community_louvain.best_partition(causal_graph.to_undirected())

# Print clusters
clusters = {}
for node, cluster in partition.items():
    clusters.setdefault(cluster, []).append(node)

print("Discovered Clusters:", clusters)


