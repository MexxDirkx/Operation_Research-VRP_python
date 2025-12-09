import random
import math
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

INPUT_XLSX = "OR 4/excel/newspaper problem instance.xlsx"
OUTPUT_XLSX = "solution.xlsx"
K = 4
random.seed(46)

# --- Data Reading & Distance ---

def read_instance(path):
    df = pd.read_excel(path)
    names = df["location"].astype(str).tolist()
    coords = list(zip(df["xcoord"].astype(float), df["ycoord"].astype(float)))
    return names, coords

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def centroid(coords):
    """Return arithmetic mean (centroid) of a list of (x,y) coords."""
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    return (sum(xs) / len(xs), sum(ys) / len(ys))

def geometric_median(coords, eps=1e-5, max_iter=1000):
    """Compute geometric median using Weiszfeld's algorithm."""
    x, y = centroid(coords)
    for _ in range(max_iter):
        num_x = 0.0
        num_y = 0.0
        denom = 0.0
        for (xi, yi) in coords:
            dx = x - xi
            dy = y - yi
            dist = math.hypot(dx, dy)
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

# --- Routing Logic ---

def build_quadrant_routes(coords, depot_idx=0, split_x=None, split_y=None):
    """Assign customers to 4 quadrants based on split point."""
    if split_x is None or split_y is None:
        cx, cy = centroid(coords)
        split_x = cx if split_x is None else split_x
        split_y = cy if split_y is None else split_y

    q_routes = [[] for _ in range(4)]

    for idx, (x, y) in enumerate(coords):
        if idx == depot_idx:
            continue
        # Strict inequalities can be adjusted, just ensuring coverage
        if x <= split_x and y >= split_y:
            q_routes[0].append(idx)
        elif x <= split_x and y < split_y:
            q_routes[1].append(idx)
        elif x > split_x and y >= split_y:
            q_routes[2].append(idx)
        else:
            q_routes[3].append(idx)

    routes = []
    for group in q_routes:
        routes.append([depot_idx] + group)
    return routes

def reorder_route_nearest_neighbor(route, coords):
    if len(route) <= 2:
        return route[:]
    depot = route[0]
    remaining = route[1:]
    ordered = [depot]
    current = depot
    while remaining:
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

def route_distance(route, coords):
    total = 0
    for i in range(len(route) - 1):
        total += manhattan(coords[route[i]], coords[route[i+1]])
    return total

def two_opt_single_route(route, coords):
    improved = True
    best_route = route[:]
    while improved:
        improved = False
        best_dist = route_distance(best_route, coords)
        for i in range(1, len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
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

def two_opt_all_routes(routes, coords):
    return [two_opt_single_route(r, coords) for r in routes]

def sa_single_route(route, coords, T0=100.0, alpha=0.995, iters=5000, min_T=1e-3):
    if len(route) <= 2:
        return route[:]
    
    curr_route = route[:]
    curr_dist = route_distance(curr_route, coords)
    best_route = curr_route[:]
    best_dist = curr_dist
    T = T0

    for _ in range(iters):
        if T < min_T:
            break
        i = random.randint(1, len(curr_route) - 2)
        j = random.randint(i + 1, len(curr_route) - 1)
        
        neighbor = curr_route[:]
        neighbor[i:j+1] = reversed(neighbor[i:j+1])
        neigh_dist = route_distance(neighbor, coords)
        
        delta = neigh_dist - curr_dist
        if delta < 0 or random.random() < math.exp(-delta / T):
            curr_route = neighbor
            curr_dist = neigh_dist
            if curr_dist < best_dist:
                best_route = curr_route[:]
                best_dist = curr_dist
        T *= alpha
    return best_route

def sa_all_routes(routes, coords, T0=100.0, alpha=0.995, iters=5000, min_T=1e-3):
    improved = []
    for r in routes:
        improved.append(sa_single_route(r, coords, T0=T0, alpha=alpha, iters=iters, min_T=min_T))
    return improved

# --- Visualization & Export ---

def visualize_routes(names, coords, routes, depot_idx=0, title="Routes", use_manhattan=True):
    plt.figure(figsize=(10,8))
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    plt.scatter(xs, ys, s=20, zorder=2, color='lightgray')
    for i, (x, y) in enumerate(coords):
        plt.text(x+0.02, y+0.02, str(i), fontsize=8)

    dx, dy = coords[depot_idx]
    plt.scatter([dx], [dy], s=120, edgecolor="k", facecolor="gold", zorder=3)
    plt.text(dx+0.05, dy+0.05, f"Depot ({depot_idx})", fontsize=9, weight="bold")

    colors = plt.cm.tab10(range(K))
    for k, r in enumerate(routes, start=1):
        color = colors[k-1]
        xs_r = [coords[i][0] for i in r]
        ys_r = [coords[i][1] for i in r]
        plt.scatter(xs_r, ys_r, s=60, color=color, zorder=4, edgecolor='white', linewidth=1)
        for i in range(len(r) - 1):
            x1, y1 = coords[r[i]]
            x2, y2 = coords[r[i+1]]
            if use_manhattan:
                plt.plot([x1, x2], [y1, y1], color=color, linewidth=2, alpha=0.7, zorder=1)
                plt.plot([x2, x2], [y1, y2], color=color, linewidth=2, alpha=0.7, zorder=1, label=f"Boy {k}" if i == 0 else "")
            else:
                plt.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7, zorder=1, label=f"Boy {k}" if i == 0 else "")

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
        for seq, cust_idx in enumerate(r[1:], start=1):
            rows.append([k, seq, cust_idx])
    pd.DataFrame(rows, columns=["Newspaper boy","Sequence number","Customer number"]).to_excel(out_path, index=False)

# --- Wrapper to run full optimization sequence ---

def solve_scenario(coords, split_x, split_y, depot_idx=0):
    """Runs the full optimization pipeline for a specific split point."""
    # 1. Split
    routes = build_quadrant_routes(coords, depot_idx=depot_idx, split_x=split_x, split_y=split_y)
    # 2. NN
    routes = reorder_all_routes_nearest_neighbor(routes, coords)
    # 3. 2-opt
    routes = two_opt_all_routes(routes, coords)
    # 4. SA
    routes = sa_all_routes(routes, coords, T0=1e4, alpha=0.999, iters=50000, min_T=1e-3)
    # 5. 2-opt clean up
    routes = two_opt_all_routes(routes, coords)
    
    return routes

# --- Main ---

if __name__ == "__main__":
    assert Path(INPUT_XLSX).exists(), f"File not found: {INPUT_XLSX}"
    names, coords = read_instance(INPUT_XLSX)
    depot_idx = 0

    # 1. Determine Candidates
    cx, cy = centroid(coords)
    
    # Calculate dataset bounds to determine a reasonable random spread
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    range_x = max(xs) - min(xs)
    range_y = max(ys) - min(ys)
    
    # Factor to perturb center (e.g., within 10% of the total map width/height)
    spread_factor = 0.1 

    candidate_centers = []
    # Candidate 1: The exact centroid
    candidate_centers.append((cx, cy))
    
    # Candidates 2-10: Random perturbations around centroid
    for _ in range(9):
        rx = cx + random.uniform(-range_x * spread_factor, range_x * spread_factor)
        ry = cy + random.uniform(-range_y * spread_factor, range_y * spread_factor)
        candidate_centers.append((rx, ry))

    print(f"Testing {len(candidate_centers)} center points...")

    # 2. Iterate and find best Min-Max solution
    best_routes = None
    best_max_dist = float('inf')
    best_center = None
    
    for i, (sx, sy) in enumerate(candidate_centers):
        # Run optimization
        current_routes = solve_scenario(coords, sx, sy, depot_idx)
        
        # metrics
        dists = [route_distance(r, coords) for r in current_routes]
        max_d = max(dists)
        total_d = sum(dists)
        
        print(f"  Test {i+1}: Center({sx:.1f}, {sy:.1f}) -> Max Route: {max_d:.2f} (Total: {total_d:.2f})")
        
        # Check if this minimizes the longest route
        if max_d < best_max_dist:
            best_max_dist = max_d
            best_routes = current_routes
            best_center = (sx, sy)

    print("-" * 40)
    print(f"Best solution found with Center ({best_center[0]:.2f}, {best_center[1]:.2f})")
    print(f"Minimised Maximum Route Distance: {best_max_dist:.2f}")
    
    total_distance = sum(route_distance(route, coords) for route in best_routes)
    print(f"Total Distance of this solution: {total_distance:.2f}")

    for k, route in enumerate(best_routes, start=1):
        dist = route_distance(route, coords)
        print(f"  Route {k}: {dist:.2f} (stops: {len(route)-1})")

    export_solution_excel(best_routes, OUTPUT_XLSX)
    print(f"Solution exported to: {OUTPUT_XLSX}")

    visualize_routes(names, coords, best_routes, depot_idx=depot_idx, 
                     title=f"Best Split Center ({best_center[0]:.1f}, {best_center[1]:.1f}) - MinMax: {best_max_dist:.2f}", 
                     use_manhattan=True)