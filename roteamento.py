from evo import Evo
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import islice

# PARAMETERS
num_points = 20
max_dist = 40
num_paths = 5
num_iteractions = 200
num_pop = 1000
mut_chance = 0.1


def k_shortest_paths(G, source, target, k, weight=None):
    return list(
        islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
    )


def draw_graph(G, pos, draw_weight=False):
    # Colors
    color_map = ['#1f78b4'] * len(G.nodes)
    color_map[0] = 'g'

    # add axis
    fig, ax = plt.subplots()
    # nx.draw(G, pos=pos, node_color='b', ax=ax)
    nx.draw(G, pos=pos, node_size=200, ax=ax, node_color=color_map)  # draw nodes and edges
    nx.draw_networkx_labels(G, pos=pos)  # draw node labels/names
    if draw_weight:
        # # draw edge weights
        labels = nx.get_edge_attributes(G, 'weight')
        labels = {key : round(labels[key], 2) for key in labels}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)
    plt.axis("on")
    ax.set_xlim(-1.5, 100*1.1)
    ax.set_ylim(-1.5, 100*1.1)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    # plt.show()

# Create points
nodes = np.random.rand(num_points, 2) * 100
nodes[0] = np.array([0, 0])
# nodes[num_points-1] = np.array([100, 100])

nodes = np.load('nodes.npy')
# np.save('nodes.npy', nodes)

edges = []
for i in range(len(nodes)):
    for j in range(i+1, len(nodes)):
        dist = np.linalg.norm(nodes[i] - nodes[j])
        if dist < max_dist:
            edges.append((i, j, dist))

# Create graph
G = nx.Graph()
G.add_weighted_edges_from(edges)

# Plot graph
points = list(map(tuple, nodes))
pos = {i: point for i, point in enumerate(points)}
draw_graph(G, points)

# K Shortest Paths
k_short_paths = []
for i in range(1, num_points):
    s_path = k_shortest_paths(G, 0, i, num_paths, "weight")
    k_short_paths.append(s_path)

k_short_paths = np.array(k_short_paths)

evolution = Evo(k_short_paths, num_paths, num_pop, mut_chance)
best, grafico = evolution.fit(num_iteractions)

# Plot do gráfico
fig, ax = plt.subplots()
plt.plot(grafico)

# Teste print 
fig, ax = plt.subplots()
SG=nx.Graph(name="mamaco")
routes = []
for i, path in enumerate(best):
    s_path = k_short_paths[i, path]
    routes.append(s_path)

print('Paths:')
print(routes)

edges = []
for r in routes:
    route_edges = [(r[n],r[n+1]) for n in range(len(r)-1)]
    SG.add_nodes_from(r)
    SG.add_edges_from(route_edges)
    edges.append(route_edges)

nx.draw_networkx_nodes(SG,pos=pos)
nx.draw_networkx_labels(SG,pos=pos)
colors = [tuple(np.random.rand(3)) for _ in range(num_points-1)]
linewidths = [1+n/2 for n in range(num_points-1)]
for ctr, edgelist in enumerate(edges):
    nx.draw_networkx_edges(SG,pos=pos,edgelist=edgelist,edge_color = colors[ctr], width=linewidths[ctr])


    # # # Subgraph
    # # s_path = nx.shortest_path(G, 0, i, "weight")
    # sub_edges = [(s_path[i-1], s_path[i]) for i in range(1, len(s_path))]
    # SG = G.edge_subgraph(sub_edges)
    # draw_graph(SG, points, True)

plt.show()