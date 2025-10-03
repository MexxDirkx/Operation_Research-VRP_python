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

def multi_route_nearest_neighbor(coords, K=4, depot_idx=0):
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
            tail = routes[k][-1]
            # dichtstbijzijnde klant voor route k
            closest_c, closest_d = None, float("inf")
            for c in unvisited:
                d = manhattan(coords[tail], coords[c])
                if d < closest_d:
                    closest_c, closest_d = c, d
            # bewaar kandidaat van deze route
            if closest_c is not None:
                cand = (closest_d, k, closest_c)
                if best is None or cand[0] < best[0]:
                    best = cand

        # voeg globaal dichtstbijzijnde kandidaat toe
        _, k, c = best
        routes[k].append(c)
        unvisited.remove(c)

    return routes

def visualize_routes(names, coords, routes, depot_idx=0, title="Routes"):
    """Quick plot: each route as a polyline, depot highlighted."""
    plt.figure(figsize=(7,6))

    # plot customers
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    plt.scatter(xs, ys, s=20, zorder=2)
    for i, (x, y) in enumerate(coords):
        plt.text(x+0.02, y+0.02, str(i), fontsize=8)

    # highlight depot
    dx, dy = coords[depot_idx]
    plt.scatter([dx], [dy], s=120, edgecolor="k", facecolor="gold", zorder=3)
    plt.text(dx+0.05, dy+0.05, f"Depot ({depot_idx})", fontsize=9, weight="bold")

    # draw each route
    for k, r in enumerate(routes, start=1):
        xs_r = [coords[i][0] for i in r]
        ys_r = [coords[i][1] for i in r]
        plt.plot(xs_r, ys_r, marker="o", linewidth=1.5, label=f"Boy {k}")

    plt.title(title)
    plt.axis("equal")
    plt.xlabel("x"); plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    assert Path(INPUT_XLSX).exists(), f"Bestand niet gevonden: {INPUT_XLSX}"
    names, coords = read_instance(INPUT_XLSX)

    routes = multi_route_nearest_neighbor(coords, K=K, depot_idx=0)
    export_solution_excel(routes, OUTPUT_XLSX)
    print(f"Gereed. Oplossing weggeschreven naar: {OUTPUT_XLSX}")

    # Visualiseer de routes
    routes = multi_route_nearest_neighbor(coords, K=K, depot_idx=0)
    export_solution_excel(routes, OUTPUT_XLSX)
    visualize_routes(names, coords, routes, depot_idx=0, title="Multi-route Nearest Neighbor")

