"""
SUMO Random Network Generator Using netconvert
-----------------------------
How to use:
1. Ensure the 'SAVE_LOCATION' variable is set to where you want to save the generated network files.
2. Ensure the 'SUMO_HOME' variable is set to your SUMO directory.
2. Run this script
3. It will automatically build the nodes, edges, and the final 'random.net.xml' network file.
"""

import math
import os
import random
import subprocess

# Set the folder path where SUMO is installed
SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
# Set the path to the tool that builds the road network
NETCONVERT_BINARY = os.path.join(SUMO_HOME, "bin", "netconvert.exe")
# Set the folder path where the generated files will be saved
SAVE_LOCATION = r"C:\Users\thoma\OneDrive - UWE Bristol\Group Project"


# Function to check the rotation of three points
def ccw(a, b, c):
    # Check if points are in counter-clockwise order
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


# Function to check if two lines cross each other
def lines_intersect(a, b, c, d):
    # Check if line segment ab intersects with line segment cd
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


# Function to check how close a point is to a line
def dist_to_segment(px, py, x1, y1, x2, y2):
    # Calculate the shortest distance from a point to a line
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    return math.hypot(px - closest_x, py - closest_y)


# Function to create the network
def generate_random_network(num_nodes: int = 20, seed: int = 42) -> str:
    # Set a random seed
    rng = random.Random(seed)

    # Create a list to hold the main junctions
    junctions = []
    # Create a list to hold the connecting roads
    roads = []
    # Create a list to track every single node point
    all_nodes = []
    # Keep track of roads that have already been connected
    already_connected = set()
    
    # Track the exact coordinates of every node
    node_coords = {}
    # Track the roads built, avoids crossing them with new roads
    built_segments = []
    # Track the roundabout locations
    roundabout_zones = []

    # Set the X coordinate
    current_x = 0.0
    # Set the Y coordinate
    current_y = 0.0
    # Give each node a unique ID number
    node_id = 0

    # Loop through to create all the main junctions
    for _ in range(num_nodes):
        # Pick a random angle to decide the direction of the next road
        angle = rng.uniform(0, 2 * math.pi)
        # Pick a distance, can be adjusted to make the network more compact or more spread out
        distance = rng.uniform(150, 350)
        
        # Calculate the new X coordinate based on the angle and distance
        current_x += math.cos(angle) * distance
        # Calculate the new Y coordinate based on the angle and distance
        current_y += math.sin(angle) * distance
        # Add random variation to the X position for a realistic layout
        current_x += rng.uniform(-30, 30)
        # Add random variation to the Y position for a realistic layout
        current_y += rng.uniform(-30, 30)

        # Set a 15% chance to build a roundabout instead of a normal junction
        if rng.random() < 0.15:
            # Randomly select a size for the roundabout
            roundabout_size = rng.choices(["small", "medium", "large"], weights=[40, 40, 20])[0]

            # Assign properties based on the chosen size
            if roundabout_size == "small":
                radius = rng.uniform(14, 18)
                # Defines the number of exits
                num_arms = rng.choices([3, 4], weights=[20, 80])[0]
            elif roundabout_size == "medium":
                radius = rng.uniform(22, 30)
                # Defines the number of exits
                num_arms = rng.choices([4, 5], weights=[70, 30])[0]
            else:
                radius = rng.uniform(35, 45)
                # Defines the number of exits
                num_arms = rng.choices([4, 5, 6], weights=[40, 40, 20])[0]

            # Register this roundabout so roads do not cross it
            roundabout_zones.append((current_x, current_y, radius))

            # Create a temporary list to track the connection points on the circle
            r_nodes = []
            # Create a temporary list to track the outward exits
            exit_nodes = []
            # Create a list to track the angles of the connection points
            angles = []

            # Loop to generate the nodes and exits
            for i in range(num_arms):
                # Calculate the angle
                angle_rad = -(2 * math.pi * i) / num_arms
                # Save the angle to the list
                angles.append(angle_rad)

                # Find the X and Y coordinates on the edge of the circle
                rx = current_x + math.cos(angle_rad) * radius
                ry = current_y + math.sin(angle_rad) * radius

                # Save the point to the main list and dictionary
                all_nodes.append((node_id, rx, ry))
                node_coords[node_id] = (rx, ry)
                # Save the ID of the perimeter point
                r_nodes.append(node_id)
                p_id = node_id
                # Increase the node ID counter by 1
                node_id += 1

                # Build a road straight out from the exit
                exit_dist = radius + 60.0
                ex = current_x + math.cos(angle_rad) * exit_dist
                ey = current_y + math.sin(angle_rad) * exit_dist

                # Save the exit point to the main list and dictionary
                all_nodes.append((node_id, ex, ey))
                node_coords[node_id] = (ex, ey)
                # Add the exit node to the junctions list 
                junctions.append((node_id, ex, ey))
                # Save the ID of the exit point
                exit_nodes.append(node_id)
                e_id = node_id
                # Increase the node ID counter by 1
                node_id += 1

                # Connect the perimeter to the exit
                roads.append((p_id, e_id, f"stub_{roundabout_size}", ""))
                # Mark this road as connected
                already_connected.add(tuple(sorted([p_id, e_id])))
                # Save the road segment so other roads cannot cross it
                built_segments.append((node_coords[p_id], node_coords[e_id]))

            # Loop to connect the points together and draw the curve
            for i in range(num_arms):
                # Get the ID of the current point
                start_node = r_nodes[i]
                # Get the ID of the next point, looping back to the start
                end_node = r_nodes[(i + 1) % num_arms]

                # Get the starting and ending angles for the curve
                start_angle = angles[i]
                end_angle = angles[(i + 1) % num_arms]
                
                # Adjust the final angle so the math draws the circle
                if i == num_arms - 1:
                    end_angle -= 2 * math.pi

                # Create an empty list to hold the curve coordinates
                shape_points = []
                # Set how many points to draw along the curve
                steps = 10
                
                # Loop to calculate the points along the road
                for j in range(steps + 1):
                    # Find the angle for this specific step along the curve
                    a = start_angle + (end_angle - start_angle) * (j / steps)
                    # Calculate the coordinates for the step
                    px = current_x + math.cos(a) * radius
                    py = current_y + math.sin(a) * radius
                    # Add the formatted coordinates to the shape list
                    shape_points.append(f"{px:.2f},{py:.2f}")

                # Join all the coordinate points together into a string
                shape_str = " ".join(shape_points)

                # Save the roundabout with its size and shape
                roads.append((start_node, end_node, f"roundabout_{roundabout_size}", shape_str))

            # Prevent inner points from connecting to each other
            for a_id in r_nodes:
                for b_id in r_nodes:
                    already_connected.add(tuple(sorted([a_id, b_id])))
                    
            # Prevent exits of the same roundabout from connecting to each other
            for e1 in exit_nodes:
                for e2 in exit_nodes:
                    already_connected.add(tuple(sorted([e1, e2])))

        # If the random chance fails, build a junction
        else:
            # Save the junction to the main node list and dictionary
            all_nodes.append((node_id, current_x, current_y))
            node_coords[node_id] = (current_x, current_y)
            # Save the junction to the main junctions list
            junctions.append((node_id, current_x, current_y))
            # Increase the node ID counter by 1
            node_id += 1

    # Assume the first junction is already part of the connected map
    connected_nodes = {0}

    # Loop through the remaining junctions to link them to the network
    for i in range(1, len(junctions)):
        # Create an empty list to store distances
        distances = []
        # Check the distance against every junction that is already connected
        for j in connected_nodes:
            # Calculate the distance between the two junctions
            dist = math.dist(
                (junctions[i][1], junctions[i][2]),
                (junctions[j][1], junctions[j][2])
            )
            # Save the distance and the target junction ID
            distances.append((dist, j))
            
        # Sort the distances to find the closest junction
        distances.sort()
        
        # Get the node ID and coordinates for the starting point
        node_i = junctions[i][0]
        point_a = node_coords[node_i]

        # Loop through the nearest nodes until a route is found
        for _, j in distances:
            # Get the node ID and coordinates for the junction
            node_j = junctions[j][0]
            point_b = node_coords[node_j]
            # Group the IDs to check if they are already connected
            pair = tuple(sorted([node_i, node_j]))

            # Skip if they are already connected
            if pair in already_connected:
                connected_nodes.add(i)
                break

            # Assume the route is valid
            route_is_valid = True

            # Check if this route cuts through a roundabout
            for cx, cy, r in roundabout_zones:
                # Add a 10-metre buffer to the radius to ensure clearance
                if dist_to_segment(cx, cy, point_a[0], point_a[1], point_b[0], point_b[1]) < (r + 10):
                    route_is_valid = False
                    break
                    
            if not route_is_valid:
                continue

            # Check if this route crosses over any existing routes
            for seg_a, seg_b in built_segments:
                # If they share a junction point it is valid
                if point_a == seg_a or point_a == seg_b or point_b == seg_a or point_b == seg_b:
                    continue
                # If the routes intersect stop the route
                if lines_intersect(point_a, point_b, seg_a, seg_b):
                    route_is_valid = False
                    break

            # If the route is safe and does not cross anything
            if route_is_valid:
                # Add the road to the list
                roads.append((node_i, node_j, "normal", ""))
                already_connected.add(pair)
                # Register the road to protect it from future crossovers
                built_segments.append((point_a, point_b))
                # Mark this junction as successfully connected
                connected_nodes.add(i)
                break

    # Add extra roads to create loops
    for i in range(len(junctions)):
        distances = []
        for j in range(len(junctions)):
            # Skip if comparing the junction to itself
            if i == j:
                continue
            # Calculate the distance
            dist = math.dist(
                (junctions[i][1], junctions[i][2]),
                (junctions[j][1], junctions[j][2])
            )
            # Save the distance
            distances.append((dist, j))
            
        # Sort the list to find the closest junctions
        distances.sort()

        # Choose to add extra roads, weight can be adjusted to change density
        extra_routes = rng.choices([0, 1], weights=[95, 5])[0]
        # Keep track of how many extra connections have been made
        routes_built = 0

        # Loop through the closest junctions until the quota is filled
        for _, j in distances:
            if routes_built >= extra_routes:
                break

            # Get the node IDs and coordinates
            node_i = junctions[i][0]
            node_j = junctions[j][0]
            point_a = node_coords[node_i]
            point_b = node_coords[node_j]
            pair = tuple(sorted([node_i, node_j]))
            
            if pair in already_connected:
                continue

            # Assume the route is valid until proven otherwise
            route_is_valid = True

            # Check if this route cuts through a roundabout
            for cx, cy, r in roundabout_zones:
                if dist_to_segment(cx, cy, point_a[0], point_a[1], point_b[0], point_b[1]) < (r + 10):
                    route_is_valid = False
                    break
                    
            if not route_is_valid:
                continue

            # Check if this route crosses any existing roads
            for seg_a, seg_b in built_segments:
                if point_a == seg_a or point_a == seg_b or point_b == seg_a or point_b == seg_b:
                    continue
                if lines_intersect(point_a, point_b, seg_a, seg_b):
                    route_is_valid = False
                    break

            # If the route is safe and does not cross anything
            if route_is_valid:
                roads.append((node_i, node_j, "normal", ""))
                already_connected.add(pair)
                built_segments.append((point_a, point_b))
                routes_built += 1

    # Set the file path for the nodes XML file
    nodes_file = os.path.join(SAVE_LOCATION, "network.nod.xml")
    # Open the file to write the data
    with open(nodes_file, "w") as f:
        # Write the opening XML tag
        f.write("<nodes>\n")
        # Loop through all generated nodes
        for nid, x, y in all_nodes:
            # Write the XML line for each node with its coordinates
            f.write(f'    <node id="n{nid}" x="{x:.0f}" y="{y:.0f}"/>\n')
        # Write the closing XML tag
        f.write("</nodes>\n")

    # Set the file path for the edges XML file
    edges_file = os.path.join(SAVE_LOCATION, "network.edg.xml")
    # Open the file to write the data
    with open(edges_file, "w") as f:
        # Write the opening XML tag
        f.write("<edges>\n")
        # Loop through all generated roads
        for i, (start, end, road_category, shape_str) in enumerate(roads):

            # Create the shape attribute text if a curve shape was provided
            shape_attr = f' shape="{shape_str}"' if shape_str else ""

            # Check if the road is the circular section of a roundabout
            if road_category.startswith("roundabout"):
                # Assign the number of lanes based on the roundabout size, will need to be changed to be randomised more
                if road_category == "roundabout_small":
                    lanes = 1
                elif road_category == "roundabout_large":
                    lanes = 3
                else:
                    lanes = 2

                # Set the speed limit to 30 mph, converted to metres per second
                speed_mps = 30 * 0.44704
                # Give the circulating road the highest priority
                priority = 100
                
                # Write the one-way road to the XML file, including the shape attribute
                f.write(f'    <edge id="road{i}a" from="n{start}" to="n{end}" '
                        f'priority="{priority}" numLanes="{lanes}" '
                        f'speed="{speed_mps:.2f}"{shape_attr}/>\n')

            # Check if the road is an immediate exit stub attached to a roundabout
            elif road_category.startswith("stub"):
                # Ensure small roundabouts do not have huge 2-lane exits squeezing into them
                if "small" in road_category:
                    lanes = 1
                else:
                    # Medium and large roundabouts can handle 1 or 2 lane exits
                    lanes = rng.choice([1, 2])
                
                # Set a standard speed for approaching a roundabout
                speed_mps = 30 * 0.44704
                # Give the approach a higher priority than standard roads
                priority = 75

                # Write the direction going away from the roundabout
                f.write(f'    <edge id="road{i}a" from="n{start}" to="n{end}" '
                        f'priority="{priority}" numLanes="{lanes}" '
                        f'speed="{speed_mps:.2f}"/>\n')
                # Write the direction going towards the roundabout to allow entry
                f.write(f'    <edge id="road{i}b" from="n{end}" to="n{start}" '
                        f'priority="{priority}" numLanes="{lanes}" '
                        f'speed="{speed_mps:.2f}"/>\n')

            # Define the rules for standard roads
            else:
                # Pick a road type
                road_type = rng.choices(
                    ["local", "b_road", "a_road"],
                    weights=[60, 30, 10]
                )[0]

                # If it is an A-road
                if road_type == "a_road":
                    # Set lanes, can be changed to adjust network
                    lanes = rng.choice([1, 2])
                    # Set the speed limit, can be changed to adjust network
                    speed_mph = rng.choice([40, 50, 60])
                # If it is a B-road
                elif road_type == "b_road":
                    lanes = 1
                    speed_mph = rng.choice([30, 40])
                # If it is a local road
                else:
                    lanes = 1
                    speed_mph = rng.choice([20, 30])

                # Convert the chosen speed limit from mph to metres per second
                speed_mps = speed_mph * 0.44704
                # Set standard roads to a lower priority than roundabouts
                priority = 50

                # Write the road direction going forward
                f.write(f'    <edge id="road{i}a" from="n{start}" to="n{end}" '
                        f'priority="{priority}" numLanes="{lanes}" '
                        f'speed="{speed_mps:.2f}"/>\n')
                # Write the road direction going backward to allow two-way traffic
                f.write(f'    <edge id="road{i}b" from="n{end}" to="n{start}" '
                        f'priority="{priority}" numLanes="{lanes}" '
                        f'speed="{speed_mps:.2f}"/>\n')

        # Write the closing XML tag
        f.write("</edges>\n")

    # Set the file path for the network file
    net_file = os.path.join(SAVE_LOCATION, "random.net.xml")
    
    # Define the commands to build the network using SUMO
    cmd = [
        # The tool that builds the network
        NETCONVERT_BINARY,
        # Point to the nodes file
        "--node-files", nodes_file,
        # Point to the edges file
        "--edge-files", edges_file,
        # Set the output destination
        "--output-file", net_file,
        # Keep traffic on the left to follow UK rules
        "--lefthand",
        # Fix any sharp or unnatural corners
        "--geometry.min-radius.fix",
        # Enable the software to apply correct roundabout right-of-way rules
        "--roundabouts.guess",
        # Add pedestrian crossings
        "--crossings.guess",
        # Prevent cars from doing U-turns in the middle of junctions
        "--no-turnarounds",
        # Add traffic lights where appropriate
        "--tls.guess",
        # Use the standard UK lane width
        "--default.lanewidth", "3.65",
        # Smooth out the visual detail of the corners
        "--junctions.corner-detail", "5",
    ]

    # Run the command to build the network
    result = subprocess.run(cmd, capture_output=True)

    # Check if the program encountered an error
    if result.returncode != 0:
        # Print the standard error heading
        print("STDERR:")
        # Print the error messages
        print(result.stderr.decode())
        # Stop the script and show an error message
        raise RuntimeError("netconvert crashed and didn't build the network.")

    # Return the path to the finished file
    return net_file


# Check if the script is being run directly
if __name__ == "__main__":
    # Generate the map
    net = generate_random_network(num_nodes=20, seed=42)
    # Print the final save location
    print("Saved to:", net)