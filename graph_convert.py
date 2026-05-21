import os
import sys
import networkx as nx

# Set the folder path where SUMO is installed
SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"

# Add SUMO's Python tools to the path so we can use sumolib
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.path.append(os.path.join(SUMO_HOME, "tools"))

# Import sumolib after
import sumolib

# Set the folder where networks are saved
SCENARIO_FOLDER = r"C:\Users\thoma\OneDrive - UWE Bristol\Group Project\Scenarios"


# Define a function to loop through and process all networks
def extract_all_graphs():
    print(f"Scanning folder: {SCENARIO_FOLDER}\n")

    # Loop through every file in the folder
    for filename in os.listdir(SCENARIO_FOLDER):
        
        # Only trigger if the file is a finished SUMO network
        if filename.endswith(".net.xml"):
            print(f"Reading network: {filename}")
            
            # Construct the path to open the file
            full_path = os.path.join(SCENARIO_FOLDER, filename)
            
            # Load the SUMO network
            net = sumolib.net.readNet(full_path)
            
            # Create an empty graph
            graph = nx.DiGraph()
            
            # Loop through roads and add them to the graph
            for road in net.getEdges():
                graph.add_edge(road.getFromNode().getID(), road.getToNode().getID())
                
            # Create a file name to save the new graph
            clean_name = filename.replace(".net.xml", "_graph.gml")
            save_path = os.path.join(SCENARIO_FOLDER, clean_name)
            
            # Save the graph in GML format
            nx.write_gml(graph, save_path)
            
            # Print a success message
            print(f"  -> Saved {graph.number_of_nodes()} junctions and {graph.number_of_edges()} roads to '{clean_name}'\n")

    print("Finished extracting all graphs.")


# Check if the script is being run directly
if __name__ == "__main__":
    # Start the extraction process
    extract_all_graphs()