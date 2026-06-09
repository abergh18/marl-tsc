"""Generate a small SUMO grid for MARL traffic-signal-control experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import runpy
from pathlib import Path
import os
import subprocess
import sys


DEFAULT_TRAFFIC_LIGHT_IDS = ("B1", "B2", "C1", "C2")# TODO: Extract dynamically from the net file instead of hardcoding these IDs. This is just to match the MAPPO notebook for now, but ideally the generator should be more flexible and not require manual updates to the traffic light IDs.


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
        grid_number: int = 4,
        lane_number: int = 1,
        traffic_light_ids: Sequence[str] = DEFAULT_TRAFFIC_LIGHT_IDS,
        trip_begin: int = 0,
        trip_end: int = 1000,
        trip_period: int = 2,
        seed: int = 42,
    ) -> None:
        """Create a SUMO grid network, traffic demand, routes, and config."""

        self.output_dir = Path(output_dir)
        self.network_filename = network_filename
        self.trips_filename = trips_filename
        self.routes_filename = routes_filename
        self.config_filename = config_filename

        self.grid_number = grid_number
        self.lane_number = lane_number
        self.traffic_light_ids = tuple(traffic_light_ids)

        self.trip_begin = trip_begin
        self.trip_end = trip_end
        self.trip_period = trip_period
        self.seed = seed

        sumo_home = os.environ.get("SUMO_HOME")
        if not sumo_home:
            raise EnvironmentError(
                "SUMO_HOME is not set. Install SUMO and set SUMO_HOME before generating simulations."
            )

        self.random_trips_file = Path(sumo_home) / "tools" / "randomTrips.py"

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

        if not self.traffic_light_ids:
            raise ValueError("traffic_light_ids must contain at least one junction ID.")

        command = [
            "netgenerate",
            "--grid",
            f"--grid.number={self.grid_number}",
            f"--default.lanenumber={self.lane_number}",
            "--tls.layout",
            "incoming",
            "--tls.set",
            ",".join(self.traffic_light_ids),
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

        if not self.random_trips_file.exists():
            raise FileNotFoundError(f"randomTrips.py not found: {self.random_trips_file}")

        tools_path = str(self.random_trips_file.parent)
        if tools_path not in sys.path:
            sys.path.insert(0, tools_path)

        old_argv = sys.argv[:]
        try:
            sys.argv = [
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
            runpy.run_path(str(self.random_trips_file), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise subprocess.CalledProcessError(int(exc.code), sys.argv) from exc
        finally:
            sys.argv = old_argv

        return self.routes_file

    def generate_config(self) -> Path:
        """Generate config.sumocfg so SUMO can run the network and routes."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        config_content = f"""<configuration>
    <input>
        <net-file value="{self.network_file.name}"/>
        <route-files value="{self.routes_file.name}"/>
    </input>

    <time>
        <begin value="{self.trip_begin}"/>
        <end value="{self.trip_end}"/>
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
