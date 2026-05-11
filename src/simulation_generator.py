"""Utilities for generating SUMO simulation files.

This module intentionally does not contain PettingZoo or reinforcement-learning
logic. It only creates the files SUMO needs: network, trips, routes, and config.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import subprocess
import sys


@dataclass
class SimulationPaths:
    """Paths created by SimulationGenerator."""

    output_dir: Path
    network_file: Path
    trips_file: Path
    routes_file: Path
    config_file: Path


class SimulationGenerator:
    def __init__(
        self,
        output_dir: str | Path = "outputs",
        network_filename: str = "network.net.xml",
        trips_filename: str = "trips.trips.xml",
        routes_filename: str = "routes.rou.xml",
        config_filename: str = "config.sumocfg",
        network_type: str = "grid",
        grid_number: int = 3,
        lane_number: int = 1,
        guess_tls: bool = True,
        trip_begin: int = 0,
        trip_end: int = 3600,
        trip_period: int = 2,
        seed: int = 42,
    ) -> None:
        """Create SUMO networks and traffic demand files.

        Args:
            output_dir: Directory where output files will be created.
            network_filename: Name of the generated SUMO network file.
            trips_filename: Name of the generated SUMO trips file.
            routes_filename: Name of the generated SUMO routes file.
            config_filename: Name of the generated SUMO config file.
            network_type: netgenerate network type. This project starts with grid.
            grid_number: Number of roads in each direction for a grid network.
            lane_number: Number of lanes per road.
            guess_tls: Whether SUMO should create guessed traffic lights.
            trip_begin: First simulation second when trips can be generated.
            trip_end: Last simulation second when trips can be generated.
            trip_period: Average seconds between generated vehicles.
            seed: Random seed used by SUMO's randomTrips.py.
        """

        self.output_dir = Path(output_dir)
        self.network_filename = network_filename
        self.trips_filename = trips_filename
        self.routes_filename = routes_filename
        self.config_filename = config_filename

        self.network_type = network_type
        self.grid_number = grid_number
        self.lane_number = lane_number
        self.guess_tls = guess_tls

        self.trip_begin = trip_begin
        self.trip_end = trip_end
        self.trip_period = trip_period
        self.seed = seed

        sumo_home = os.environ.get("SUMO_HOME")
        if not sumo_home:
            raise EnvironmentError(
                "SUMO_HOME is not set. Install SUMO and set SUMO_HOME before generating simulations."
            )

        self.sumo_home = Path(sumo_home)
        self.random_trips_file = self.sumo_home / "tools" / "randomTrips.py"

    @property
    def network_file(self) -> Path:
        return self.output_dir / self.network_filename

    @property
    def trips_file(self) -> Path:
        return self.output_dir / self.trips_filename

    @property
    def routes_file(self) -> Path:
        return self.output_dir / self.routes_filename

    @property
    def config_file(self) -> Path:
        return self.output_dir / self.config_filename

    @property
    def paths(self) -> SimulationPaths:
        return SimulationPaths(
            output_dir=self.output_dir,
            network_file=self.network_file,
            trips_file=self.trips_file,
            routes_file=self.routes_file,
            config_file=self.config_file,
        )

    def run_command(self, command: list[str]) -> None:
        """Run an external SUMO command and fail loudly on errors."""
        subprocess.run(command, check=True)

    def generate_network(self) -> Path:
        """Generate the SUMO network XML file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.network_type != "grid":
            raise NotImplementedError(
                "This starter project currently supports network_type='grid'. "
                "Add spider/rand options once the grid version works."
            )

        command = [
            "netgenerate",
            "--grid",
            f"--grid.number={self.grid_number}",
            f"--tls.guess={str(self.guess_tls).lower()}",
            f"--default.lanenumber={self.lane_number}",
            "-o",
            str(self.network_file),
        ]

        self.run_command(command)
        return self.network_file

    def generate_trips(self) -> Path:
        """Generate trips and route files for the generated network."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.network_file.exists():
            raise FileNotFoundError(
                f"Network file does not exist: {self.network_file}. Run generate_network() first."
            )

        command = [
            sys.executable,
            str(self.random_trips_file),
            "-n",
            str(self.network_file),
            "-o",
            str(self.trips_file),
            "-r",
            str(self.routes_file),
            "--begin",
            str(self.trip_begin),
            "--end",
            str(self.trip_end),
            "--period",
            str(self.trip_period),
            "--seed",
            str(self.seed),
        ]

        self.run_command(command)
        return self.routes_file

    def generate_config(self) -> Path:
        """Generate config.sumocfg so SUMO can run the network and routes."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        config_content = f"""<configuration>
    <input>
        <net-file value=\"{self.network_file.name}\"/>
        <route-files value=\"{self.routes_file.name}\"/>
    </input>

    <time>
        <begin value=\"{self.trip_begin}\"/>
        <end value=\"{self.trip_end}\"/>
    </time>
</configuration>
"""

        self.config_file.write_text(config_content, encoding="utf-8")
        return self.config_file

    def generate_all(self) -> SimulationPaths:
        """Generate every file required to run the SUMO simulation."""
        self.generate_network()
        self.generate_trips()
        self.generate_config()
        return self.paths
