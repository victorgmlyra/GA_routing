from evo import Evo
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import islice


def k_shortest_paths(G, source, target, k, weight=None):
    return list(
        islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
    )

def draw_graph_with_subpaths(best, k_short_paths, pos):
    # Teste print 
    fig, ax = plt.subplots()
    SG=nx.Graph(name="graph")
    routes = []
    for i, path in enumerate(best):
        s_path = k_short_paths[i, path]
        routes.append(s_path)


    for i in range(0,len(routes)):
        print('Path for node',i+1,': ',routes[i])

    edges = []
    for r in routes:
        route_edges = [(r[n],r[n+1]) for n in range(len(r)-1)]
        SG.add_nodes_from(r)
        SG.add_edges_from(route_edges)
        edges.append(route_edges)

    nx.draw_networkx_nodes(SG,pos=pos)
    nx.draw_networkx_labels(SG,pos=pos)
    colors = [tuple(np.random.rand(3)) for _ in range(len(pos)-1)]
    linewidths = [1+n/2 for n in range(len(pos)-1)]
    for ctr, edgelist in enumerate(edges):
        nx.draw_networkx_edges(SG,pos=pos,edgelist=edgelist,edge_color = colors[ctr], width=linewidths[ctr])

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

def build_graph(params, load=True):
    # Create points
    if load:
        nodes = np.load('nodes.npy')
    else:
        nodes = np.random.rand(params['num_points'], 2) * 100
        nodes[0] = np.array([0, 0])
    # nodes[num_points-1] = np.array([100, 100])

    # np.save('nodes.npy', nodes)

    edges = []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            dist = np.linalg.norm(nodes[i] - nodes[j])
            if dist < params['max_dist']:
                edges.append((i, j, dist))

    # Create graph
    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    # Plot graph
    points = list(map(tuple, nodes))
    pos = {i: point for i, point in enumerate(points)}
    draw_graph(G, points, True)

    return G, pos

def find_all_k_short_paths(G, params):
    # K Shortest Paths
    k_short_paths = []
    for i in range(1, params['num_points']):
        s_path = k_shortest_paths(G, 0, i, params['num_paths'], "weight")
        k_short_paths.append(s_path)

    k_short_paths = np.array(k_short_paths)
    return k_short_paths

def find_optimal(k_short_paths, pos, params):
    evolution = Evo(k_short_paths, params['num_paths'], params['num_pop'], pos, params['mut_chance'], params['fitness_alg'])
    return evolution.fit(params['num_iteractions'])

if __name__ == '__main__':
    # PARAMETERS
    params = {
        # Graph building
        'num_points': 20,
        'load_graph': True,
        'max_dist': 40,
        # K shortest paths
        'num_paths': 5,
        # GA params
        'num_pop': 100,
        'num_iteractions': 100,
        'mut_chance': 0.1,
        'fitness_alg': 'lifetime',    # energy, lifetime
        # Plot
        'plot': True
    }

    # Build graph
    G, pos = build_graph(params, params['load_graph'])
    k_short_paths = find_all_k_short_paths(G, params)

    # Run genetic algorithm
    num_runs = 1    # Change to run multiple times
    all_lifetimes, all_sum_energies = [], []
    for i in range(num_runs):
        best, evol_fit, lifetime, sum_energies, node = find_optimal(k_short_paths, pos, params)
        print('Best: ', best, ' Lifetime: ', lifetime, ' for node: ', node, ' Sum energies: ',sum_energies)
        print()
        all_lifetimes.append(lifetime)
        all_sum_energies.append(sum_energies)

    # Plot do gráfico
    fig, ax = plt.subplots()
    plt.plot(evol_fit)
    # Plot graph with subpaths
    draw_graph_with_subpaths(best, k_short_paths, pos)
    if params['plot'] == True:
        plt.show()

    # Print
    print('\nMean best lifetimes: {:.3f}'.format(np.mean(np.array(all_lifetimes))))
    print('Mean best sum energies: {:.3f}'.format(np.mean(np.array(all_sum_energies))))
