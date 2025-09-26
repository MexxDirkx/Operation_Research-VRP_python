import matplotlib.pyplot as plt
import pandas as pd

def read_coordinates_from_excel(file_path):
    """Read coordinates from Excel file with columns: name, x, y"""
    df = pd.read_excel(file_path)
    coordinations = [[row['x'], row['y']] for _, row in df.iterrows()]
    print(f"Read {len(coordinations)} coordinates from Excel file")
    return coordinations

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

def calculate_tour_distance(path, distance_matrix):
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
    current_distance = calculate_tour_distance(current_path, distance_matrix)
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
                new_distance = calculate_tour_distance(new_path, distance_matrix)
                
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

def main():
    """Main function to run the TSP solver."""
    # Check if Excel file exists
    excel_file = "OR 4/excel/berlin52.xlsx"
    coordinations = read_coordinates_from_excel(excel_file)

    print(f"Working with {len(coordinations)} locations")

    # Precompute distances
    print("=== PRECOMPUTING DISTANCES ===")
    distance_matrix = precompute_distance_matrix(coordinations)
    
    # Step 1: Solve TSP using nearest neighbor heuristic
    print("=== NEAREST NEIGHBOR HEURISTIC ===")
    current_loc = 0
    initial_path, initial_distance = nearest_neighbor_tsp(coordinations, distance_matrix, current_loc)
    
    print(f"Initial tour path: {initial_path}")
    print(f"Initial total distance: {initial_distance:.3f}")
    
    # Step 2: Improve the tour using 2-opt
    print("\n=== 2-OPT IMPROVEMENT ===")
    improved_path, improved_distance = two_opt_improve(coordinations, initial_path, distance_matrix)
    
    print(f"\nFinal tour path: {improved_path}")
    print(f"Final total distance: {improved_distance:.3f}")
    print(f"Improvement: {initial_distance - improved_distance:.3f} ({((initial_distance - improved_distance) / initial_distance * 100):.1f}%)")
    
    # Step 3: Visualize both tours
    plt.figure(figsize=(12, 5))
    
    # Plot initial tour
    plt.subplot(1, 2, 1)
    visualize_tour(coordinations, initial_path, initial_distance, " (Nearest Neighbor)")
    
    # Plot improved tour
    plt.subplot(1, 2, 2)
    visualize_tour(coordinations, improved_path, improved_distance, " (After 2-opt)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()