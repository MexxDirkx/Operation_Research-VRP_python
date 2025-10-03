import pandas as pd
import os

def excel_to_tsp(excel_file, output_file="output.tsp", name="TSP_INSTANCE"):
    """
    Convert Excel coordinates (columns: name, x, y) to TSPLIB .tsp format.
    """
    if not os.path.exists(excel_file):
        raise FileNotFoundError(f"Excel file not found: {excel_file}")

    df = pd.read_excel(excel_file)
    if not {"name", "x", "y"}.issubset(df.columns):
        raise ValueError("Excel must contain columns: name, x, y")

    cities = df[["name", "x", "y"]].values.tolist()

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"NAME: {name}\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {len(cities)}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (_, x, y) in enumerate(cities, start=1):
            f.write(f"{i} {x} {y}\n")
        f.write("EOF\n")

    print(f"TSP file exported to {output_file}")

if __name__ == "__main__":
    excel_to_tsp("OR 4/excel/rd100.xlsx", "rd100.tsp", name="rd100")
