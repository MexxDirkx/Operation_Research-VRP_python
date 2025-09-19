import matplotlib.pyplot as plt
import pandas as pd
import os

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def precompute_distance_matrix(coordinations):
    """Precompute all pairwise distances."""
    n = len(coordinations)
    matrix = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i+1, n):
            dist = distance(coordinations[i], coordinations[j])
            matrix[i][j] = dist
            matrix[j][i] = dist  # Symmetric
    
    return matrix

def nearest_neighbor_tsp(coordinations, distance_matrix, start_loc=0):
    """
    Solve TSP using nearest neighbor heuristic.
    
    Args:
        coordinations: List of [x, y] coordinates
        start_loc: Starting location index
    
    Returns:
        tuple: (path, total_distance)
    """
    locations = list(range(len(coordinations)))
    current_loc = start_loc
    unvisited = locations.copy()
    unvisited.remove(current_loc)
    
    total_distance = 0
    path = [current_loc]
    
    while unvisited:
        closest = float('inf')
        next_loc = None
        
        # Find the closest location
        for i in unvisited:
            dist = distance_matrix[i][current_loc]
            if dist < closest:
                closest = dist
                next_loc = i
        
        # Move to the closest location
        total_distance += closest
        path.append(next_loc)
        
        current_loc = next_loc
        unvisited.remove(next_loc)
    
    # Return to start
    total_distance += distance_matrix[current_loc][path[0]]
    path.append(path[0])
    
    return path, total_distance

def calculate_tour_distance_fast(path, distance_matrix):
    """Fast tour distance using precomputed matrix."""
    total = 0.0
    for i in range(len(path) - 1):
        total += distance_matrix[path[i]][path[i + 1]]
    return total


def two_opt_swap(path, i, j):
    """
    Perform a 2-opt swap on a tour path.
    
    This reverses the order of cities between positions i and j.
    For example: [0,1,2,3,4,0] with i=1, j=3 becomes [0,3,2,1,4,0]
    
    Args:
        path: Current tour path
        i, j: Positions to swap between (i < j)
    
    Returns:
        list: New path with 2-opt swap applied
    """
    # Create new path: start + reversed middle section + end
    new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
    return new_path

def two_opt_improve(coordinations, initial_path, distance_matrix):
    """
    Improve a tour using 2-opt local search.
    
    The 2-opt algorithm works by:
    1. Taking the current best tour
    2. For every pair of edges, try removing them and reconnecting differently
    3. If this creates a shorter tour, keep the improvement
    4. Repeat until no more improvements can be found
    
    Args:
        coordinations: List of [x, y] coordinates
        initial_path: Starting tour path
    
    Returns:
        tuple: (improved_path, improved_distance)
    """
    current_path = initial_path.copy()
    current_distance = calculate_tour_distance_fast(current_path, distance_matrix)
    improved = True
    
    print("Starting 2-opt improvement...")
    print(f"Initial distance: {current_distance:.3f}")
    
    iteration = 0
    while improved:
        improved = False
        iteration += 1
        
        # Try all possible 2-opt swaps
        # We exclude the last city since it's the same as the first (return to start)
        for i in range(1, len(current_path) - 2):
            for j in range(i + 1, len(current_path) - 1):
                # Create new tour with 2-opt swap
                new_path = two_opt_swap(current_path, i, j)
                new_distance = calculate_tour_distance_fast(new_path, distance_matrix)
                
                # If we found an improvement, keep it
                if new_distance < current_distance:
                    current_path = new_path
                    current_distance = new_distance
                    improved = True
                    print(f"Iteration {iteration}: Found improvement! New distance: {current_distance:.3f}")
                    break  # Start over with the improved tour
            
            if improved:
                break  # Start over from the beginning
    
    print(f"2-opt completed after {iteration} iterations")
    print(f"Final distance: {current_distance:.3f}")
    
    return current_path, current_distance



def visualize_tour(coordinations, path, total_distance, title_suffix=""):
    """
    Visualize the TSP tour.
    
    Args:
        coordinations: List of [x, y] coordinates
        path: Tour path as list of location indices
        total_distance: Total tour distance
    """
    # Plot all locations
    for i in range(len(coordinations)):
        x = coordinations[i][0]
        y = coordinations[i][1]
        plt.scatter(x, y, c="k")
        plt.text(x + 0.02, y + 0.02, str(i), ha="center")
    
    # Plot the tour path
    tour_x = []
    tour_y = []
    for i in path:
        tour_x.append(coordinations[i][0])
        tour_y.append(coordinations[i][1])
    plt.plot(tour_x, tour_y, "b-")
    
    plt.title("Number of locations: " + str(len(coordinations)) +
              ", Tour distance: " + str(round(total_distance, 2)) + title_suffix)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")

def read_coordinates_from_excel(file_path):
    """Read coordinates from Excel file with columns: name, x, y"""
    try:
        df = pd.read_excel(file_path)
        coordinations = [[row['x'], row['y']] for _, row in df.iterrows()]
        print(f"Read {len(coordinations)} coordinates from Excel file")
        return coordinations
    
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

def _tour_distance_from_cycle(cycle, distance_matrix):
    """Distance of a tour given as a cycle (no repeated start at end)."""
    total = 0.0
    n = len(cycle)
    for i in range(n):
        a = cycle[i]
        b = cycle[(i + 1) % n]
        total += distance_matrix[a][b]
    return total

def three_opt_improve(coordinations, initial_path, distance_matrix, verbose=True):
    """
    Improve a tour using a simple 3-opt local search.

    Notes:
      - Works on a 'cycle' (path without the last repeated start node).
      - Tries 7 standard reconnection cases for segments (i..j-1) and (j..k-1).
      - First-improvement strategy: as soon as a better tour is found, restart scan.
    """
    # Work on a cycle (drop the last repeated node)
    cycle = initial_path[:-1]
    n = len(cycle)
    if n < 6:  # 3-opt needs enough nodes to matter
        return initial_path, calculate_tour_distance_fast(initial_path, distance_matrix)

    best_cycle = cycle[:]
    best_dist = _tour_distance_from_cycle(best_cycle, distance_matrix)

    if verbose:
        print("Starting 3-opt improvement...")
        print(f"Initial (pre-3opt) distance: {best_dist:.3f}")

    improved = True
    iteration = 0

    while improved:
        improved = False
        iteration += 1

        # i < j < k define the three breakpoints between segments
        for i in range(1, n - 3):                # keep 0 fixed as a convenient anchor
            for j in range(i + 1, n - 2):
                for k in range(j + 1, n - 1):
                    # Split into four parts: P | A | B | S
                    # where:
                    #   prefix = best_cycle[:i]
                    #   A      = best_cycle[i:j]
                    #   B      = best_cycle[j:k]
                    #   suffix = best_cycle[k:]
                    prefix = best_cycle[:i]
                    A = best_cycle[i:j]
                    B = best_cycle[j:k]
                    suffix = best_cycle[k:]

                    # Enumerate 7 standard reconnections (excluding identity):
                    candidates = [
                        prefix + A[::-1] + B + suffix,       # Case 1
                        prefix + A + B[::-1] + suffix,       # Case 2
                        prefix + A[::-1] + B[::-1] + suffix, # Case 3
                        prefix + B + A + suffix,             # Case 4 (swap A,B)
                        prefix + B[::-1] + A + suffix,       # Case 5
                        prefix + B + A[::-1] + suffix,       # Case 6
                        prefix + B[::-1] + A[::-1] + suffix  # Case 7
                    ]

                    # Evaluate and keep the first improving candidate
                    for cand in candidates:
                        cand_dist = _tour_distance_from_cycle(cand, distance_matrix)
                        if cand_dist + 1e-12 < best_dist:
                            best_cycle = cand
                            best_dist = cand_dist
                            improved = True
                            if verbose:
                                print(f"Iteration {iteration}: 3-opt improvement -> {best_dist:.3f}")
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    if verbose:
        print(f"3-opt completed after {iteration} iterations")
        print(f"Final (post-3opt) distance: {best_dist:.3f}")

    # Return to closed path form (repeat start at end)
    improved_path = best_cycle + [best_cycle[0]]
    improved_distance = calculate_tour_distance_fast(improved_path, distance_matrix)
    return improved_path, improved_distance



def main():
    """Main function to run the TSP solver with 3-opt."""
    # Check if Excel file exists
    excel_file = "OR 4/excel/rd100.xlsx"
    
    if os.path.exists(excel_file):
        coordinations = read_coordinates_from_excel(excel_file)
        if coordinations is None:
            coordinations = [[565.0, 575.0], [25.0, 185.0], [345.0, 750.0], [945.0, 685.0],
                             [845.0, 655.0], [880.0, 660.0], [25.0, 230.0], [525.0, 1000.0],
                             [580.0, 1175.0], [650.0, 1130.0], [1605.0, 620.0], [1220.0, 580.0],
                             [1465.0, 200.0]]
    else:
        # Example coordinates
        coordinations = [[565.0, 575.0], [25.0, 185.0], [345.0, 750.0], [945.0, 685.0],
                         [845.0, 655.0], [880.0, 660.0], [25.0, 230.0], [525.0, 1000.0],
                         [580.0, 1175.0], [650.0, 1130.0], [1605.0, 620.0], [1220.0, 580.0],
                         [1465.0, 200.0]]
    
    print(f"Working with {len(coordinations)} locations")

    # Precompute distances
    print("=== PRECOMPUTING DISTANCES ===")
    distance_matrix = precompute_distance_matrix(coordinations)
    
    # Step 1: Nearest Neighbor
    print("=== NEAREST NEIGHBOR HEURISTIC ===")
    current_loc = 0
    nn_path, nn_distance = nearest_neighbor_tsp(coordinations, distance_matrix, current_loc)
    print(f"Initial tour path (NN): {nn_path}")
    print(f"Initial total distance (NN): {nn_distance:.3f}")
    
    # Step 2: 2-Opt improvement
    print("\n=== 2-OPT IMPROVEMENT ===")
    two_opt_path, two_opt_distance = two_opt_improve(coordinations, nn_path, distance_matrix)
    print(f"\nAfter 2-opt: {two_opt_distance:.3f} (Δ = {nn_distance - two_opt_distance:.3f}, {((nn_distance - two_opt_distance)/nn_distance*100):.1f}%)")
    
    # Step 3: 3-Opt improvement
    print("\n=== 3-OPT IMPROVEMENT ===")
    three_opt_path, three_opt_distance = three_opt_improve(coordinations, two_opt_path, distance_matrix)
    print(f"\nAfter 3-opt: {three_opt_distance:.3f} (Δ vs 2-opt = {two_opt_distance - three_opt_distance:.3f}, {((two_opt_distance - three_opt_distance)/two_opt_distance*100):.1f}%)")
    print(f"Total improvement vs NN: {nn_distance - three_opt_distance:.3f} ({((nn_distance - three_opt_distance)/nn_distance*100):.1f}%)")

    # Visualize all three tours
    plt.figure(figsize=(18, 5))
    
    # NN
    plt.subplot(1, 3, 1)
    visualize_tour(coordinations, nn_path, nn_distance, " (Nearest Neighbor)")
    
    # 2-Opt
    plt.subplot(1, 3, 2)
    visualize_tour(coordinations, two_opt_path, two_opt_distance, " (After 2-opt)")
    
    # 3-Opt
    plt.subplot(1, 3, 3)
    visualize_tour(coordinations, three_opt_path, three_opt_distance, " (After 3-opt)")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()