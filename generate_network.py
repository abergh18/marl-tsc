"""
Network Generator
-----------------------
This script automatically downloads and builds different types of UK road networks.
It uses real map data from several locations to create varied scenarios.
"""

import os
import time
import random
import subprocess
import requests
import osmnx as ox

# Set the folder path where SUMO is installed
SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"


# Define a function to convert the map into a SUMO network file
def build_sumo_network(osm_path: str, net_path: str):
    # Set the path to the tool that builds the road network
    netconvert = os.path.join(SUMO_HOME, "bin", "netconvert.exe")

    # Define the commands to build the network
    cmd = [
        netconvert,
        "--osm-files", osm_path,
        "--output-file", net_path,
        # Keep traffic on the left
        "--lefthand",
        # Remove unnecessary internal junctions
        "--no-internal-links",  
        # Group close junctions together to simplify the map
        "--junctions.join",
        # Group close edges together
        "--edges.join", 
        # Simplify the physical shapes of the roads
        "--geometry.remove",
        # Remove small geometric details under 5 metres
        "--geometry.min-dist", "5",
        # Add roundabouts where appropriate
        "--roundabouts.guess",
        # Prevent cars from doing U-turns
        "--no-turnarounds",
        # Add traffic lights where appropriate
        "--tls.guess",
        # Group traffic lights that are close together
        "--tls.join",
        # Set roads to have 1 lane by default
        "--default.lanenumber", "1",
        # Set the default lane width
        "--default.lanewidth", "3.2",
    ]

    # Run the command to build the network
    result = subprocess.run(cmd, capture_output=True)
    
    # Check if the program encountered an error
    if result.returncode != 0:
        # Print an error message if it failed
        print(f"  Warning: netconvert failed for {osm_path}")
        print(result.stderr.decode())
    else:
        # Print a success message
        print(f"  Successfully created SUMO network: {net_path}")


# Define a function to loop through locations and generate maps for each
def generate_scenarios(num_locations=None):
    # Set the folder path where the generated files will be saved
    save_location = r"C:\Users\thoma\OneDrive - UWE Bristol\Group Project\Scenarios"

    locations = [
        "London, UK", "Birmingham, UK", "Manchester, UK", "Glasgow, UK",
        "Newcastle, UK", "Sheffield, UK", "Liverpool, UK", "Leeds, UK",
        "Bristol, UK", "Belfast, UK", "Edinburgh, UK", "Leicester, UK",
        "Coventry, UK", "Bradford, UK", "Cardiff, UK", "Nottingham, UK",
        "Kingston upon Hull, UK", "Stoke-on-Trent, UK", "Wolverhampton, UK",
        "Plymouth, UK", "Derby, UK", "Southampton, UK", "Reading, UK",
        "Swansea, UK", "Aberdeen, UK", "Preston, UK", "Sunderland, UK",
        "Norwich, UK", "Chester, UK", "Carlisle, UK", "Exeter, UK",
        "Gloucester, UK", "Worcester, UK", "York, UK", "Lancaster, UK",
        "Newport, UK", "Dundee, UK", "Portsmouth, UK", "Oxford, UK",
        "Cambridge, UK", "Bath, UK", "Milton Keynes, UK", "Lincoln, UK",
        "Durham, UK", "Canterbury, UK", "St Albans, UK", "Winchester, UK",
        "Salisbury, UK", "Lichfield, UK", "Hereford, UK", "Stirling, UK",
        "Inverness, UK", "Truro, UK", "Ely, UK", "Ripon, UK",
        "Brighton, UK", "Bournemouth, UK", "Middlesbrough, UK", "Blackpool, UK",
        "Bolton, UK", "Stockport, UK", "Slough, UK", "Watford, UK",
        "Rotherham, UK", "Cheltenham, UK", "Chelmsford, UK", "Colchester, UK",
        "Ipswich, UK", "Peterborough, UK", "Northampton, UK", "Luton, UK",
        "Swindon, UK", "Southend-on-Sea, UK", "Telford, UK", "Shrewsbury, UK",
        "Guildford, UK", "Basingstoke, UK", "Eastbourne, UK", "Hastings, UK",
        "Dover, UK", "Perth, UK", "Dunfermline, UK", "Paisley, UK",
        "East Kilbride, UK", "Ayr, UK", "Dumfries, UK", "Wrexham, UK",
        "Bangor, UK", "Carmarthen, UK", "Merthyr Tydfil, UK", "Derry, UK",
        "Lisburn, UK", "Newry, UK", "Coleraine, UK", "Ballymena, UK",
        "Taunton, UK", "Weymouth, UK", "Torquay, UK", "Harrogate, UK",
        "Halifax, UK",
    ]

    # If a number is supplied, randomly select that many locations
    if num_locations is not None:
        if num_locations > len(locations):
            raise ValueError(
                f"Requested {num_locations} locations but only {len(locations)} are available."
            )

        locations = random.sample(locations, num_locations)
    # Set the web address for the server that provides the map data
    overpass_url = "https://overpass-api.de/api/interpreter"
    # Set a user agent so the server knows what program is asking for the data
    headers = {"User-Agent": "MARL-Heterogeneous-Generator/1.0"}

    # Loop through each location in the list
    for idx, location in enumerate(locations):
        print(f"\nLocation: {location}")
        
        # Create a simple file name based on the location
        clean_filename = location.lower().replace(", uk", "").replace(" ", "_")
        
        # Try to process the location, catching any errors that happen
        try:
            # Find centre of location
            centre_lat, centre_lon = ox.geocode(location)

            # Get all roads within 3km of centre
            G = ox.graph_from_point(
                    (centre_lat, centre_lon),
                    dist=3000,
                    network_type="drive"
                )

            # Randomly select a node from the graph to be the centre of the map
            node = random.choice(list(G.nodes))

            # Get the latitude and longitude of the selected node
            lat = G.nodes[node]["y"]
            lon = G.nodes[node]["x"]
            
            # Pick a random radius to make maps different sizes
            radius = random.randint(350, 550)

            # Create the text query to ask the server for all the driving roads in that area
            query = f"""[out:xml];
            way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential"](around:{radius},{lat},{lon});
            (._;>;);out body;"""

            # Send the request to the server and download the data
            response = requests.post(overpass_url, data={'data': query}, headers=headers)
            # Check if the download failed and raise an error if it did
            response.raise_for_status()

            # Set the file path to save the map data
            osm_file = os.path.join(save_location, f"{clean_filename}.osm.xml")
            # Set the file path to save the finished SUMO network
            net_file = os.path.join(save_location, f"{clean_filename}.net.xml")

            # Open the file and write the downloaded data into it
            with open(osm_file, "w", encoding="utf-8") as file:
                file.write(response.text)
                
            # Call the function to convert the data into a SUMO map
            build_sumo_network(osm_file, net_file)

        # If an error happens, print a message and skip to the next location
        except Exception as e:
            print(f"  Error processing {location}: {e}")
            continue

        # Pause for 5 seconds so the server does not block the script for asking too fast
        print("Pausing")
        time.sleep(5)


# Check if the script is being run directly
if __name__ == "__main__":
    # Start the batch generation process
    generate_scenarios()
