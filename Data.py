# ----------------------------------------
# Network Analysis of Twitter Retweet Data
# Vashawn Robinson – INST414 Final Project
# ----------------------------------------

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from networkx.algorithms.community import greedy_modularity_communities


# ----------------------------------------
# 1. Load and Clean Data
# ----------------------------------------

# Load CSV file
df = pd.read_csv("twitter_retweet_data.csv")

# Basic cleaning
df = df.dropna(subset=["user", "retweeted_user"])
df = df.drop_duplicates(subset=["user", "retweeted_user"])

print("Data loaded.")
print(f"Rows: {len(df)}")


# ----------------------------------------
# 2. Build Directed Retweet Network
# ----------------------------------------

# Create directed graph
G = nx.DiGraph()

# Add edges: information flows from retweeted_user -> user
for _, row in df.iterrows():
    G.add_edge(row["retweeted_user"], row["user"])

print("Graph constructed.")
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")


# ----------------------------------------
# 3. Centrality Analysis
# ----------------------------------------

# Degree centrality
degree_centrality = nx.degree_centrality(G)

# PageRank (primary influence metric)
pagerank_scores = nx.pagerank(G, alpha=0.85)

# Store PageRank in DataFrame
pr_df = pd.DataFrame.from_dict(
    pagerank_scores, orient="index", columns=["pagerank"]
)


# ----------------------------------------
# 4. Categorize Users by Influence Level
# ----------------------------------------

# Create influence tiers based on PageRank
pr_df["influence_level"] = pd.qcut(
    pr_df["pagerank"],
    q=[0, 0.7, 0.95, 1.0],
    labels=["Low Influence", "Medium Influence", "High Influence"]
)

influence_counts = pr_df["influence_level"].value_counts().sort_index()


# ----------------------------------------
# 5. Plot: Influence Distribution
# ----------------------------------------

plt.figure()
plt.bar(influence_counts.index, influence_counts.values)
plt.title("Distribution of Influence Levels in the Network")
plt.xlabel("Influence Category")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.savefig("influence_distribution.png")
plt.close()

print("Saved influence_distribution.png")


# ----------------------------------------
# 6. Community Detection
# ----------------------------------------

# Convert to undirected graph for modularity-based clustering
undirected_G = G.to_undirected()

communities = list(greedy_modularity_communities(undirected_G))

# Map each node to a community
community_map = {}
for i, community in enumerate(communities):
    for node in community:
        community_map[node] = i

pr_df["community"] = pr_df.index.map(community_map)


# ----------------------------------------
# 7. Plot: Community Distribution
# ----------------------------------------

community_sizes = pr_df["community"].value_counts().sort_index()

plt.figure()
plt.pie(
    community_sizes.values,
    labels=[f"Community {i}" for i in community_sizes.index],
    autopct="%1.1f%%"
)
plt.title("User Distribution Across Detected Communities")
plt.tight_layout()
plt.savefig("community_distribution.png")
plt.close()

print("Saved community_distribution.png")


# ----------------------------------------
# 8. Identify Bridge Nodes
# ----------------------------------------

bridge_scores = {}

for node in G.nodes():
    node_community = community_map.get(node)
    cross_links = 0

    for neighbor in G.neighbors(node):
        if community_map.get(neighbor) != node_community:
            cross_links += 1

    bridge_scores[node] = cross_links

bridge_df = (
    pd.DataFrame.from_dict(
        bridge_scores, orient="index", columns=["cross_community_links"]
    )
    .sort_values(by="cross_community_links", ascending=False)
    .head(10)
)


# ----------------------------------------
# 9. Plot: Bridge Nodes
# ----------------------------------------

plt.figure()
plt.bar(bridge_df.index, bridge_df["cross_community_links"])
plt.title("Top Bridge Nodes by Cross-Community Connections")
plt.xlabel("User")
plt.ylabel("Number of Cross-Community Links")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("bridge_nodes.png")
plt.close()

print("Saved bridge_nodes.png")


# ----------------------------------------
# 10. Optional: Output Top Influential Users
# ----------------------------------------

top_users = pr_df.sort_values(by="pagerank", ascending=False).head(10)
print("\nTop 10 Influential Users (PageRank):")
print(top_users)

print("\nAnalysis complete.")
