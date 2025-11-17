import random
import pandas as pd

random.seed(42)

def generate_locations_xlsx(
    n_locations=120,
    x_min=0, x_max=3000,
    y_min=0, y_max=3000,
    depot_x=1200, depot_y=1200,
    output_file="OR 4/excel/locations.xlsx",
    seed=None
):
    """
    Generate an Excel file with columns: location, xcoord, ycoord.
    Row 1 is 'depot', others are 1..n_locations with random coordinates.
    """
    if seed is not None:
        random.seed(seed)

    rows = []

    # Depot row
    rows.append({"location": "depot", "xcoord": depot_x, "ycoord": depot_y})

    # Customer/location rows
    for i in range(1, n_locations + 1):
        x = random.randint(x_min, x_max)
        y = random.randint(y_min, y_max)
        rows.append({"location": i, "xcoord": x, "ycoord": y})

    df = pd.DataFrame(rows, columns=["location", "xcoord", "ycoord"])
    df.to_excel(output_file, index=False)
    print(f"Generated {output_file} with {n_locations} locations + depot.")

if __name__ == "__main__":
    generate_locations_xlsx()