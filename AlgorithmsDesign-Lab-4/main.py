from ant_colony_optimization import *

def main():
    test_graph = get_random_graph(10)
    it = 201
    min_length = calculate_optimal_length(test_graph)
    print("L_min =", min_length)
    print("Number of iterations =", it)
    print("Cycle:\n", ant_colony_optimization(g = test_graph, iterations = it, plot_the_graph = True, min_length = min_length))
    print("Graph")
    print(test_graph)
    
if __name__ == "__main__":
    main()
