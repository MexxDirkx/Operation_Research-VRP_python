import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import random
import math

INPUT_XLSX = "OR 4/excel/newspaper problem instance.xlsx"
OUTPUT_XLSX = "solution.xlsx"
K = 4

def read_instance(path):
    df = pd.read_excel(path)
    names = df["location"].astype(str).tolist()
    coords = list(zip(df["xcoord"].astype(float), df["ycoord"].astype(float)))
    return names, coords

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def export_solution_excel(routes, out_path):
    rows = []
    for k, r in enumerate(routes, start=1):
        # r = [depot, c1, c2, ...]  -> schrijf GEEN depot weg
        for seq, cust_idx in enumerate(r[1:], start=1):
            rows.append([k, seq, cust_idx])  # Customer number = index uit input
    pd.DataFrame(rows, columns=["Newspaper boy","Sequence number","Customer number"]).to_excel(out_path, index=False)


### NEW FUNCTIONS FOR QUADRANT-BASED ROUTING + REORDERING + 2-OPT ###
def build_quadrant_routes(coords, depot_idx=0, split_x=280, split_y=275):
    """
    Deel alle klanten in op 4 kwadranten t.o.v. (split_x, split_y).

    Let op: volgorde binnen elk kwadrant is nog niet slim (gewoon willekeurig),
    we optimaliseren die daarna met nearest-neighbor-achtig en 2-opt.
    """
    q_routes = [[] for _ in range(4)]

    for idx, (x, y) in enumerate(coords):
        if idx == depot_idx:
            continue
        if x <= split_x and y >= split_y:
            q_routes[0].append(idx)  # Q1
        elif x <= split_x and y < split_y:
            q_routes[1].append(idx)  # Q2
        elif x > split_x and y >= split_y:
            q_routes[2].append(idx)  # Q3
        else:
            q_routes[3].append(idx)  # Q4

    # prepend depot to each route (each boy starts at depot)
    routes = []
    for group in q_routes:
        routes.append([depot_idx] + group)

    return routes

def reorder_route_nearest_neighbor(route, coords):
    """
    Neem een enkele route zoals [depot, c1, c2, ...]
    en herordent de klanten (behalve depot) met een simpele greedy nearest-neighbor,
    beginnend vanaf het depot.
    """
    if len(route) <= 2:
        return route[:]  # 0 of 1 klant, niets te doen

    depot = route[0]
    remaining = route[1:]  # klanten
    ordered = [depot]

    current = depot
    while remaining:
        # pak dichtstbijzijnde
        best_i = None
        best_d = float("inf")
        for i, cust in enumerate(remaining):
            d = manhattan(coords[current], coords[cust])
            if d < best_d:
                best_d = d
                best_i = i
        nxt = remaining.pop(best_i)
        ordered.append(nxt)
        current = nxt

    return ordered

def reorder_all_routes_nearest_neighbor(routes, coords):
    return [reorder_route_nearest_neighbor(r, coords) for r in routes]
### END NEW FUNCTIONS ###


def visualize_routes(names, coords, routes, depot_idx=0, title="Routes", use_manhattan=True):
    """
    Plot routes with optional Manhattan (L-shaped) routing.
    
    Args:
        names: List of location names
        coords: List of (x, y) coordinates
        routes: List of routes, each route is a list of indices
        depot_idx: Index of the depot location
        title: Plot title
        use_manhattan: If True, draw L-shaped routes; if False, draw straight lines
    """
    plt.figure(figsize=(10,8))

    # plot customers
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    plt.scatter(xs, ys, s=20, zorder=2, color='lightgray')
    for i, (x, y) in enumerate(coords):
        plt.text(x+0.02, y+0.02, str(i), fontsize=8)

    # highlight depot
    dx, dy = coords[depot_idx]
    plt.scatter([dx], [dy], s=120, edgecolor="k", facecolor="gold", zorder=3)
    plt.text(dx+0.05, dy+0.05, f"Depot ({depot_idx})", fontsize=9, weight="bold")

    # draw each route
    colors = plt.cm.tab10(range(K))
    for k, r in enumerate(routes, start=1):
        color = colors[k-1]
        
        # Plot customer points on this route
        xs_r = [coords[i][0] for i in r]
        ys_r = [coords[i][1] for i in r]
        plt.scatter(xs_r, ys_r, s=60, color=color, zorder=4, edgecolor='white', linewidth=1)
        
        # Draw connections between consecutive points
        for i in range(len(r) - 1):
            x1, y1 = coords[r[i]]
            x2, y2 = coords[r[i+1]]
            
            if use_manhattan:
                # Draw L-shaped route (Manhattan style)
                # Choose horizontal-then-vertical (you could also alternate or use a heuristic)
                plt.plot([x1, x2], [y1, y1], color=color, linewidth=2, alpha=0.7, zorder=1)
                plt.plot([x2, x2], [y1, y2], color=color, linewidth=2, alpha=0.7, zorder=1, 
                        label=f"Boy {k}" if i == 0 else "")
            else:
                # Draw straight line
                plt.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7, zorder=1,
                        label=f"Boy {k}" if i == 0 else "")

    plt.title(title)
    plt.axis("equal")
    plt.xlabel("x"); plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def route_distance(route, coords):
    """Bereken totale Manhattan afstand van een route."""
    total = 0
    for i in range(len(route) - 1):
        total += manhattan(coords[route[i]], coords[route[i+1]])
    return total

def two_opt_single_route(route, coords, depot_idx=0):
    """
    Voer 2-opt uit op één route.
    De depot (eerste punt) blijft vast.
    """
    improved = True
    best_route = route[:]
    
    while improved:
        improved = False
        best_dist = route_distance(best_route, coords)
        
        # Probeer alle mogelijke segment-omdraaiingen
        for i in range(1, len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                # Maak nieuwe route door segment [i:j+1] om te draaien
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                new_dist = route_distance(new_route, coords)
                
                if new_dist < best_dist:
                    best_route = new_route
                    best_dist = new_dist
                    improved = True
                    break
            if improved:
                break
    
    return best_route

def two_opt_all_routes(routes, coords, depot_idx=0):
    """Pas 2-opt toe op alle routes."""
    improved_routes = []
    for route in routes:
        improved = two_opt_single_route(route, coords, depot_idx)
        improved_routes.append(improved)
    return improved_routes

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
def get_neighbor(route, distance_matrix):
    """
    Generate a neighbor solution using various neighborhood operations.
    Returns the new tour and the change in distance.
    """
    neighbor = route.copy()
    operation = random.choice(['swap', 'two_opt', 'insert'])
    
    if operation == 'swap':
        # Swap two random cities (excluding start/end which are the same)
        i = random.randint(1, len(neighbor) - 2)
        j = random.randint(1, len(neighbor) - 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    
    elif operation == 'two_opt':
        # Random 2-opt move
        i = random.randint(1, len(neighbor) - 3)
        j = random.randint(i + 1, len(neighbor) - 2)
        neighbor = two_opt_swap(neighbor, i, j)
    
    elif operation == 'insert':
        # Remove a city and insert it elsewhere
        if len(neighbor) > 3:  # Need at least 3 cities for insertion
            remove_idx = random.randint(1, len(neighbor) - 2)
            city = neighbor.pop(remove_idx)
            insert_idx = random.randint(1, len(neighbor) - 1)
            neighbor.insert(insert_idx, city)
    
    return neighbor

def simulated_annealing(initial_tour, route_distance, initial_temp=500000, cooling_rate=0.6, min_temp=0.1):
    """
    Improve a tour using simulated annealing.
    
    Args:
        initial_tour: Starting tour path
        distance_matrix: Precomputed distance matrix
        initial_temp: Starting temperature
        cooling_rate: Temperature reduction factor (0 < cooling_rate < 1)
        min_temp: Minimum temperature to stop
    
    Returns:
        tuple: (best_tour, best_distance, temperatures, distances)
    """
    current_tour = initial_tour.copy()
    current_distance = route_distance(current_tour)
    
    best_tour = current_tour.copy()
    best_distance = current_distance
    
    temperature = initial_temp
    iteration = 0
    
    # For tracking progress
    temperatures = []
    distances = []
    
    print("Starting simulated annealing...")
    print(f"Initial distance: {current_distance:.3f}")
    print(f"Initial temperature: {temperature}")
    
    while temperature > min_temp:
        iteration += 1
        
        # Generate neighbor
        neighbor_tour = get_neighbor(current_tour, route_distance(current_distance))
        neighbor_distance = route_distance(neighbor_tour)
        
        # Calculate change in distance
        delta = neighbor_distance - current_distance
        
        # Accept or reject the neighbor
        if delta < 0:  # Better solution
            current_tour = neighbor_tour
            current_distance = neighbor_distance
            
            # Check if it's the best so far
            if current_distance < best_distance:
                best_tour = current_tour.copy()
                best_distance = current_distance
                print(f"Iteration {iteration}: New best distance: {best_distance:.3f}")
        
        else:  # Worse solution - accept with probability
            probability = math.exp(-delta / temperature)
            if random.random() < probability:
                current_tour = neighbor_tour
                current_distance = neighbor_distance
        
        # Cool down
        temperature *= cooling_rate
        
        # Track progress every 100 iterations
        if iteration % 100 == 0:
            temperatures.append(temperature)
            distances.append(best_distance)
            print(f"Iteration {iteration}: Temperature: {temperature:.3f}, Best distance: {best_distance:.3f}")
    
    print(f"Simulated annealing completed after {iteration} iterations")
    print(f"Final best distance: {best_distance:.3f}")
    
    return best_tour, best_distance, temperatures, distances


if __name__ == "__main__":
    assert Path(INPUT_XLSX).exists(), f"Bestand niet gevonden: {INPUT_XLSX}"
    names, coords = read_instance(INPUT_XLSX)

    # 1. Maak kwadrantenroutes op basis
    routes = build_quadrant_routes(coords, depot_idx=0, split_x=280, split_y=275)

    # 2. Herorden met nearest-neighbor binnen elk kwadrant
    routes = reorder_all_routes_nearest_neighbor(routes, coords)

    # 3. Pas 2-opt toe op elke route
    routes = two_opt_all_routes(routes, coords, depot_idx=0)
    
    # 4 Pas Simulated Annealing toe op elke route
    improved_routes = []
    
    for k in range(K):
        routes[k], route_distance[k,coords], temps, dists = simulated_annealing(
            initial_tour=routes[k],
            route_distance=lambda r: route_distance(k, coords),
            initial_temp=500000,
            cooling_rate=0.6,
            min_temp=0.1
        )
        improved_routes.append(best_tour)

    total_distance = sum(route_distance(route, coords) for route in routes)
    print(f"\nTotale afstand van alle routes: {total_distance:.2f}")

    for k, route in enumerate(routes, start=1):
        dist = route_distance(route, coords)
        print(f"  Route {k}: {dist:.2f} (met {len(route)-1} klanten)")

    export_solution_excel(routes, OUTPUT_XLSX)
    print(f"Gereed. Oplossing weggeschreven naar: {OUTPUT_XLSX}")

    visualize_routes(names, coords, routes, depot_idx=0, 
                    title="Multi-route Nearest Neighbor + 2-Opt per route (Manhattan Routes)", 
                    use_manhattan=True)

