import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

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

def multi_route_nearest_neighbor(coords, K=4, depot_idx=0, capacity=30):
    """
    Idee:
    - We hebben K routes, elk start bij het depot.
    - In iedere stap: voor elke route zoek je de dichtstbijzijnde ONBEZOCHTE klant vanaf het LAATSTE punt van die route.
      Dat levert tot K kandidaten op. Kies daarvan de globaal dichtstbijzijnde en voeg die toe.
    - Herhaal tot alle klanten zijn toegewezen.
    """
    n = len(coords)
    routes = [[depot_idx] for _ in range(K)]
    unvisited = set(range(n)); unvisited.remove(depot_idx)
    
    while unvisited:
        best = None  # (dist, route_k, customer_c)
        for k in range(K):
            # Skip routes that are at capacity
            if len(routes[k]) - 1 >= capacity:  # -1 because depot doesn't count
                continue
                
            tail = routes[k][-1]
            # Find closest customer for route k
            closest_c, closest_d = None, float("inf")
            for c in unvisited:
                d = manhattan(coords[tail], coords[c])
                if d < closest_d:
                    closest_c, closest_d = c, d
                    
            # Save candidate from this route
            if closest_c is not None:
                cand = (closest_d, k, closest_c)
                if best is None or cand[0] < best[0]:
                    best = cand
        
        if best is None:
            print("Warning: Some customers could not be assigned - routes at capacity")
            break
            
        # Add globally closest candidate
        _, k, c = best
        routes[k].append(c)
        unvisited.remove(c)

    return routes
        
            
        
            
    
        
            
        

    return routes

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











if __name__ == "__main__":
    assert Path(INPUT_XLSX).exists(), f"Bestand niet gevonden: {INPUT_XLSX}"
    names, coords = read_instance(INPUT_XLSX)


    routes = multi_route_nearest_neighbor(coords, K=K, depot_idx=0, capacity=31)
    routes = two_opt_all_routes(routes, coords, depot_idx=0)

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
