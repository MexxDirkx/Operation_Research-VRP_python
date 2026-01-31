import random
import math
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

INPUT_XLSX = "excel/newspaper problem instance.xlsx"
OUTPUT_XLSX = "solution.xlsx"
K = 4
random.seed(46)

# Reading instance & manhattan distance
def read_instance(path):
    df = pd.read_excel(path)
    names = df["location"].astype(str).tolist()
    coords = list(zip(df["xcoord"].astype(float), df["ycoord"].astype(float)))
    return names, coords

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# Centroid and Geometric Median
def centroid(coords):
    """Return arithmetic mean (centroid) of a list of (x,y) coords."""
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    return (sum(xs) / len(xs), sum(ys) / len(ys))

def geometric_median(coords, eps=1e-5, max_iter=1000):
    """Compute geometric median using Weiszfeld's algorithm.

    Returns a point (x,y) that minimises sum of Euclidean distances to coords.
    """
    # start at centroid
    x, y = centroid(coords)
    for _ in range(max_iter):
        num_x = 0.0
        num_y = 0.0
        denom = 0.0
        for (xi, yi) in coords:
            dx = x - xi
            dy = y - yi
            dist = math.hypot(dx, dy)
            # if current estimate is exactly on a data point, that's the median
            if dist < eps:
                return (xi, yi)
            w = 1.0 / dist
            num_x += xi * w
            num_y += yi * w
            denom += w
        new_x = num_x / denom
        new_y = num_y / denom
        move = math.hypot(new_x - x, new_y - y)
        x, y = new_x, new_y
        if move < eps:
            break
    return (x, y)

# Quadrant & nearest-neighbor functions
def build_quadrant_routes(coords, depot_idx=0, split_x=None, split_y=None):
    """
    Deel alle klanten in op 4 kwadranten t.o.v. (split_x, split_y).

    Let op: volgorde binnen elk kwadrant is nog niet slim (gewoon willekeurig),
    we optimaliseren die daarna met nearest-neighbor-achtig en 2-opt.
    """
    # If no split point provided, use centroid of coordinates
    if split_x is None or split_y is None:
        cx, cy = centroid(coords)
        if split_x is None:
            split_x = cx
        if split_y is None:
            split_y = cy

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

# Two-opt functions
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

# Simulated Annealing functions
def sa_single_route(route, coords, depot_idx=0,
                    T0=100.0, alpha=0.995, iters=5000, min_T=1e-3):
    """
    Simulated annealing TSP-style improvement for ONE route.
    - Keeps depot fixed at position 0.
    - Neighbor move = reverse a random segment [i:j] (i>=1).
    - Accepts worse moves with probability exp(-Δ/T) so we can escape local minima.

    Parameters:
        T0:    initial temperature
        alpha: cooling rate per step (0<alpha<1)
        iters: number of iterations
        min_T: stop early if temperature drops below this

    Returns:
        improved route (list of node indices)
    """

    # Routes with <=2 stops (depot + 0/1 customer) can't really improve
    if len(route) <= 2:
        return route[:]

    # current solution
    curr_route = route[:]
    curr_dist = route_distance(curr_route, coords)

    # best-so-far
    best_route = curr_route[:]
    best_dist = curr_dist

    T = T0

    for _ in range(iters):
        if T < min_T:
            break

        # pick two cut points in [1 .. len-1] so depot (0) stays at front
        i = random.randint(1, len(curr_route) - 2)
        j = random.randint(i + 1, len(curr_route) - 1)

        # neighbor: reverse the segment curr_route[i:j+1]
        neighbor = curr_route[:]
        neighbor[i:j+1] = reversed(neighbor[i:j+1])

        neigh_dist = route_distance(neighbor, coords)
        delta = neigh_dist - curr_dist

        # accept if better OR with some probability if worse
        if delta < 0 or random.random() < math.exp(-delta / T):
            curr_route = neighbor
            curr_dist = neigh_dist

            if curr_dist < best_dist:
                best_route = curr_route[:]
                best_dist = curr_dist

        # cool down
        T *= alpha

    return best_route

def sa_all_routes(routes, coords, depot_idx=0,
                  T0=100.0, alpha=0.995, iters=5000, min_T=1e-3):
    """
    Run simulated annealing on each route independently.
    """
    improved = []
    for r in routes:
        improved.append(
            sa_single_route(
                r, coords, depot_idx=depot_idx,
                T0=T0, alpha=alpha, iters=iters, min_T=min_T
            )
        )
    return improved

# Visualization and export functions
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

def export_solution_excel(routes, out_path):
    rows = []
    for k, r in enumerate(routes, start=1):
        # r = [depot, c1, c2, ...]  -> schrijf GEEN depot weg
        for seq, cust_idx in enumerate(r[1:], start=1):
            rows.append([k, seq, cust_idx])  # Customer number = index uit input
    pd.DataFrame(rows, columns=["Newspaper boy","Sequence number","Customer number"]).to_excel(out_path, index=False)


def evaluate_split(coords, depot_idx, split_x, split_y):
    """
    Run the full pipeline for a given split point (split_x, split_y).

    Returns:
        routes        : list of routes
        max_len       : max route distance over the 4 routes
        total_len     : sum of all route distances (just for info)
        route_lengths : list of 4 distances
    """
    # 1. Quadrant assignment
    routes = build_quadrant_routes(coords, depot_idx=depot_idx,
                                   split_x=split_x, split_y=split_y)

    # 2. Nearest neighbor within each quadrant
    routes = reorder_all_routes_nearest_neighbor(routes, coords)

    # 3. 2-opt
    routes = two_opt_all_routes(routes, coords, depot_idx=depot_idx)

    # 4. Simulated annealing
    routes = sa_all_routes(
        routes, coords, depot_idx=depot_idx,
        T0=1e3, alpha=0.995, iters=5000, min_T=1e-3
    )

    # 5. Final 2-opt
    routes = two_opt_all_routes(routes, coords, depot_idx=depot_idx)

    # Distances
    route_lengths = [route_distance(r, coords) for r in routes]
    max_len = max(route_lengths)
    total_len = sum(route_lengths)

    return routes, max_len, total_len, route_lengths

# Main procedure
if __name__ == "__main__":
    assert Path(INPUT_XLSX).exists(), f"Bestand niet gevonden: {INPUT_XLSX}"
    names, coords = read_instance(INPUT_XLSX)

    # Choose how to pick the base split point / starting location.
    # "centroid" (arithmetic mean) / "geometric_median"
    start_method = "centroid"

    if start_method == "centroid":
        base_sx, base_sy = centroid(coords)
    elif start_method == "geometric_median":
        base_sx, base_sy = geometric_median(coords)
    else:
        raise ValueError(f"Unknown start_method: {start_method}")

    depot_idx = 0

    # Determine a step size for "around" points (tunable!)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    range_x = max(xs) - min(xs)
    range_y = max(ys) - min(ys)
    # small step relative to the instance size
    step = 0.2 * min(range_x, range_y) if min(range_x, range_y) > 0 else 1.0

    # 10 candidate split points: center + 9 around it
    # You can tweak this pattern however you like.
    offsets = [
        (0.0, 0.0),          # center
        ( step, 0.0),        # right
        (-step, 0.0),        # left
        (0.0,  step),        # up
        (0.0, -step),        # down
        ( step,  step),      # up-right
        ( step, -step),      # down-right
        (-step,  step),      # up-left
        (-step, -step),      # down-left
        (2*step, 0.0),       # a bit further right (10th point)
    ]

    candidate_centers = [
        (base_sx + dx, base_sy + dy) for (dx, dy) in offsets
    ]

    print(f"Base start method: {start_method}, base split at "
          f"({base_sx:.2f}, {base_sy:.2f}), depot index: {depot_idx}")
    print("Trying candidate split points:")

    best_routes = None
    best_center = None
    best_max_len = float("inf")
    best_total_len = None
    best_lengths = None

    for idx, (sx, sy) in enumerate(candidate_centers, start=1):
        print(f"\nCandidate {idx}: split at ({sx:.2f}, {sy:.2f})")

        routes, max_len, total_len, route_lengths = evaluate_split(
            coords, depot_idx, sx, sy
        )

        print(f"  Route lengths: {', '.join(f'{d:.2f}' for d in route_lengths)}")
        print(f"  Max route length: {max_len:.2f}")
        print(f"  Total distance:   {total_len:.2f}")

        # Our objective: minimize the maximum route length
        if max_len < best_max_len:
            best_max_len = max_len
            best_routes = routes
            best_center = (sx, sy)
            best_total_len = total_len
            best_lengths = route_lengths
            print("  -> New best solution w.r.t. max route length")

    print("\n=== Best solution over all candidate split points ===")
    print(f"Best split point: ({best_center[0]:.2f}, {best_center[1]:.2f})")
    print(f"Best max route length: {best_max_len:.2f}")
    print(f"Best total distance:   {best_total_len:.2f}")

    for k, (dist, route) in enumerate(zip(best_lengths, best_routes), start=1):
        print(f"  Route {k}: {dist:.2f} (met {len(route)-1} klanten)")

    # Export and visualize the best solution
    export_solution_excel(best_routes, OUTPUT_XLSX)
    print(f"\nGereed. Oplossing weggeschreven naar: {OUTPUT_XLSX}")

    visualize_routes(
        names, coords, best_routes, depot_idx=depot_idx,
        title="Best split point (minimizing max route length)",
        use_manhattan=True
    )
