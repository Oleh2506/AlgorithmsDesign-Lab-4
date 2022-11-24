from audioop import reverse
from xml.etree.ElementTree import tostring
import numpy as np
import random
import matplotlib.pyplot as plt

class Graph:
    def __init__(self, edge_weights, default_pheromone_level = None) -> None:

        assert edge_weights.shape[0] == edge_weights.shape[1]

        self.vertex_cardinality = edge_weights.shape[0]
        self.edge_weights = edge_weights

        if default_pheromone_level:
            self.pheromone_levels = np.full_like(edge_weights, default_pheromone_level).astype('float64')
        else:
            self.pheromone_levels = np.full_like(edge_weights, self.edge_weights.mean()).astype('float64')

    def __str__(self):

        return f'Vertex Cardinality = {str(self.vertex_cardinality)}\nEdge Weights Matrix:\n{self.edge_weights}\nPheromone Levels Matrix:\n{self.pheromone_levels}'

def cycle_length(g, cycle):

    length = 0
    i = 0

    while i < len(cycle) - 1:
        length += g.edge_weights[cycle[i]][cycle[i+1]]
        i += 1

    return length

def traverse_graph(g, initial_vertex = 0, alpha_value = 2.0, beta_value = 4.0):
    
    visited = np.asarray([False for _ in range(g.vertex_cardinality)])
    visited[initial_vertex] = True
    cycle = [initial_vertex]
    curr_vertex = initial_vertex
    path_length = 0

    for _ in range(g.vertex_cardinality - 1):
        
        jumps_neighbors = []
        jumps_values = []
        jumps = []
        for vertex in range(g.vertex_cardinality):
            if not visited[vertex]:
               pheromone_level = max(g.pheromone_levels[curr_vertex][vertex], 1e-5) 
               v = (pheromone_level**alpha_value) / (g.edge_weights[curr_vertex][vertex]**beta_value) 
               jumps.append((vertex, v))
               jumps_neighbors.append(vertex)
               jumps_values.append(v)
               
        next_vertex = random.choices(jumps_neighbors, weights = jumps_values, k = 1)[0]

        visited[next_vertex] = True
        curr_vertex = next_vertex
        cycle.append(curr_vertex)

    cycle.append(initial_vertex)
    path_length = cycle_length(g, cycle)

    return cycle, path_length

def calculate_optimal_length(g: Graph) -> np.float64:

    min_length = np.inf

    for i in range(g.vertex_cardinality):
        initial_vertex = i
        curr_vertex = initial_vertex
        
        visited = np.asarray([False for _ in range(g.vertex_cardinality)])
        visited[i] = True
        curr_length = 0

        for _ in range(g.vertex_cardinality):
            curr_min_vertex = curr_vertex
            curr_min = np.inf
            
            for vertex in range(g.vertex_cardinality):
                if not visited[vertex] and curr_min > g.edge_weights[curr_vertex][vertex]:
                    curr_min_vertex = vertex
                    curr_min = g.edge_weights[curr_vertex][curr_min_vertex]

            visited[curr_min_vertex] = True
            curr_length += g.edge_weights[curr_vertex][curr_min_vertex]
            curr_vertex = curr_min_vertex

        curr_length += g.edge_weights[curr_vertex][initial_vertex]

        # print(curr_length)

        if min_length > curr_length:
            min_length = curr_length

    return min_length

def ant_colony_optimization(g, alpha_value = 2.0, beta_value = 4.0, min_length = None, iterations = 1000, ants_per_iteration = 30, degradation_factor = .4, plot_the_graph = False):

    if not min_length:
        min_length = calculate_optimal_length(g)
        print("L_min = ", min_length)

    best_cycle = (None, np.inf)
    old_best = (None, np.inf)
    inertia = 0
    patience = 50

    x = []
    y = []

    for i in range(iterations):
        cycles = []
        cycles = [traverse_graph(g, random.randint(0, g.vertex_cardinality - 1), alpha_value, beta_value) 
                  for _ in range(ants_per_iteration)] 
        curr_best_cycle = (None, np.inf)
        best_cycle_as_for_now = best_cycle

        if best_cycle[0]:
            cycles.append(best_cycle)
            old_best = best_cycle

        g.pheromone_levels *= (1 - degradation_factor)

        for cycle, path_length in cycles: 

            if path_length < best_cycle[1]:
                best_cycle = (cycle, path_length)

            if curr_best_cycle[1] > path_length and cycle != best_cycle_as_for_now[0]:
                curr_best_cycle = (cycle, path_length)

            delta = min_length / path_length
            j = 0

            while j < len(cycle) - 1:

                g.pheromone_levels[cycle[j]][cycle[j+1]] += delta
                j += 1

        if i % 20 == 0:
            if plot_the_graph:
                x.append(i + 1)
                y.append(curr_best_cycle[1])
                # print(curr_best_cycle[1])

        if best_cycle[0]:  
            
            if old_best == best_cycle:
                inertia += 1
            else:
                inertia = 0

            if inertia > patience:
                g.pheromone_levels += g.pheromone_levels.mean()
                inertia = 0
                
    if plot_the_graph:
        plt.xlabel("Iterations") 
        plt.ylabel("Path Length") 
        plt.plot(x,y)
        plt.savefig('iterations_vs_length.pdf', format = 'pdf')
            
    return best_cycle

def get_random_graph(n, rand_min = 5, rand_max = 50):

    assert n > 1

    matrix = np.random.randint(rand_min, rand_max + 1, (n, n)).astype('float64')
    for i in range(n):
        matrix[i][i] = 0
        j = i + 1
        while j < n:
            matrix[j][i] = matrix[i][j]
            j += 1
    graph = Graph(matrix)

    return graph

