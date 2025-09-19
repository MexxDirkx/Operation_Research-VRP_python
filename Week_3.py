import random
import matplotlib.pyplot as plt

coordinations = [[0, 0], [0, 1], [0.5, 1], [0.5, 0], [0.2, 0.5], [0.3, 0.5]]
locations = range(0, len(coordinations))

def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


# current_loc = random.choice(locations)
current_loc = 0
unvisited = list(locations)
unvisited.remove(current_loc)

total_distance = 0
path = [current_loc]

while unvisited:
    closest = float('inf')
    next_loc = None

    # Find the closest location
    for i in unvisited:
        dist = distance(coordinations[i], coordinations[current_loc])
        if dist < closest:
            closest = dist
            next_loc = i

    # Move to the closest location
    total_distance += closest
    path.append(next_loc)

    current_loc = next_loc
    unvisited.remove(next_loc)

# Return to start
total_distance += distance(coordinations[current_loc], coordinations[path[0]])
path.append(path[0])

print(f"Tour path: {path}")
print(f"Total distance: {total_distance}")

# --- Visualization ---
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
          ", Tour distance: " + str(round(total_distance, 2)))
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.show()