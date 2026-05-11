"""
SUMO Random Network Generator Using netgenerate
-----------------------------
How to use:
1. Ensure the 'SAVE_LOCATION' variable is set to where you want to save the generated network files.
2. Ensure the 'SUMO_HOME' variable is set to your SUMO directory.
2. Run this script
3. It will automatically build the nodes, edges, and the final 'random.net.xml' network file.
"""

import os
import random
import subprocess

# Path to SUMO installation
SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
# Path to netgenerate
NETGENERATE_BINARY = os.path.join(SUMO_HOME, "bin", "netgenerate.exe")
# Save location
SAVE_LOCATION = "C:\\Users\\thoma\\OneDrive - UWE Bristol\\Group Project"

def generate_random_network(seed: int = 42) -> str:
    # Create file
    net_file = os.path.join(SAVE_LOCATION, f"random.net.xml")
    # Create a random generator
    rng = random.Random(seed)
    # Random number of junctions
    num_junctions = rng.randint(15, 30)
    # Random number of lanes per road
    lanes = rng.randint(1, 3)
    # Randomly choose speed limits
    speed_mph = rng.choice([30, 40, 50, 60])
    # Convert mph → meters per second
    speed = speed_mph * 0.44704

    # Create the netgenerate command
    cmd = [
        NETGENERATE_BINARY,
        # Random mode — generates mixed junction types automatically
        "--rand",
        # Number of junctions to generate
        f"--rand.iterations={num_junctions}",
        # Road length between junctions
        "--rand.min-distance=100",
        "--rand.max-distance=200",
        # Lanes and speed
        f"--default.lanenumber={lanes}",
        f"--default.speed={speed:.2f}",
        # Place traffic lights
        "--tls.guess",
        # Seed for reproducibility
        f"--seed={seed}",
        # Output file
        f"-o={net_file}",
    ]

    # Run the SUMO command
    result = subprocess.run(cmd, capture_output=True)

    # AI generated code to check for and print SUMO errors
    if result.returncode != 0:
        print(result.stderr.decode())
        raise RuntimeError("netgenerate failed")


if __name__ == "__main__":
    net = generate_random_network(seed=42)