"""Generate SUMO files for MARL traffic-signal-control experiments."""

from __future__ import annotations

import os
import runpy
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from marl_tsc.network_types import GridNetwork, NetworkType


@dataclass
class SimulationPaths:
    """Paths created by SimulationGenerator."""

    output_dir: Path
    network_file: Path
    trips_file: Path
    routes_file: Path
    config_file: Path
    traffic_light_ids: tuple[str, ...]


class SimulationGenerator:
    def __init__(
        self,
        output_dir: str | Path = "outputs",
        network_filename: str = "network.net.xml",
        trips_filename: str = "trips.trips.xml",
        routes_filename: str = "routes.rou.xml",
        config_filename: str = "config.sumocfg",
        network: NetworkType | None = None,
        traffic_light_ids: Sequence[str] | None = None,
        trip_begin: int = 0,
        trip_end: int = 1000,
        trip_period: float = 2,
        trip_fringe_factor: str | float | None = None,
        allow_fringe: bool = True,
        random_depart: bool = True,
        seed: int = 42,
        min_distance: int | None = None,
        vtypes_file: str | Path | None = None,
    ) -> None:
        """Create a SUMO network, traffic demand, routes, and config."""

        self.output_dir = Path(output_dir)
        self.network_filename = network_filename
        self.trips_filename = trips_filename
        self.routes_filename = routes_filename
        self.config_filename = config_filename

        self.network = network or GridNetwork()
        self.traffic_light_ids = tuple(traffic_light_ids) if traffic_light_ids else None

        self.trip_begin = trip_begin
        self.trip_end = trip_end
        self.trip_period = trip_period
        self.trip_fringe_factor = trip_fringe_factor
        self.allow_fringe = allow_fringe
        self.random_depart = random_depart
        self.seed = seed
        self.min_distance = min_distance
        self.vtypes_file = Path(vtypes_file) if vtypes_file else None

        sumo_home = os.environ.get("SUMO_HOME")
        if not sumo_home:
            raise EnvironmentError(
                "SUMO_HOME is not set. Install SUMO and set SUMO_HOME before generating simulations."
            )

        self.random_trips_file = Path(sumo_home) / "tools" / "randomTrips.py"

    def discover_traffic_light_ids(self) -> tuple[str, ...]:
        """Read traffic light IDs directly from the network file."""
        import sumolib
        
        net = sumolib.net.readNet(str(self.network_file))
        ids = tuple(tls.getID() for tls in net.getTrafficLights())
        
        if not ids:
            raise ValueError(f"No traffic lights found in {self.network_file}")
            
        print(f"Discovered {len(ids)} traffic lights: {list(ids)}")
        return ids

    def _random_trips_argv(
        self,
        *,
        trip_fringe_factor: str | float | None,
        allow_fringe: bool,
    ) -> list[str]:
        argv = [
            str(self.random_trips_file),
            "-n",
            str(self.network_file),
            "-o",
            str(self.trips_file),
            "--begin",
            str(self.trip_begin),
            "--end",
            str(self.trip_end),
            "--period",
            str(self.trip_period),
            "--seed",
            str(self.seed),
        ]

        if trip_fringe_factor not in (None, "", 1, "1", "1.0"):
            argv.extend(["--fringe-factor", str(trip_fringe_factor)])
            if allow_fringe:
                argv.append("--allow-fringe")

        if self.random_depart:
            argv.append("--random-depart")

        if self.min_distance is not None:
            argv.extend(["--min-distance", str(self.min_distance)])

        if self.vtypes_file is not None:
            argv.extend(["--additional-files", str(self.vtypes_file)])
            argv.extend(["--vehicle-class", "passenger"])

        argv.extend(["--edge-permission", "passenger"])
        argv.extend(["--speed-exponent", "2"])

        return argv

    def _run_random_trips(self, argv: list[str]) -> None:
        old_argv = sys.argv[:]
        try:
            sys.argv = argv
            runpy.run_path(str(self.random_trips_file), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise subprocess.CalledProcessError(int(exc.code), argv) from exc
        finally:
            sys.argv = old_argv

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
            traffic_light_ids=self.traffic_light_ids,
        )

    def run_command(self, command: list[str]) -> None:
        """Run an external SUMO command and fail loudly on errors."""
        subprocess.run(command, check=True)

    def generate_network(self) -> Path:
        """Generate the SUMO network XML file and detect its traffic lights."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Delegate network generation to the newly integrated NetworkType class
        self.network.generate(self.output_dir, self.network_file, self.run_command)

        # If traffic light IDs were not explicitly provided, detect them
        if not self.traffic_light_ids:
            self.traffic_light_ids = self.detect_traffic_light_ids()

        if not self.traffic_light_ids:
            raise ValueError(f"No traffic lights were found in {self.network_file}.")

        return self.network_file

    def detect_traffic_light_ids(self) -> tuple[str, ...]:
        """Read the generated network and return its traffic-light ids."""
        tree = ET.parse(self.network_file)
        traffic_light_ids = []
        for element in tree.iter():
            if element.tag.endswith("tlLogic"):
                traffic_light_id = element.attrib.get("id")
                if traffic_light_id and traffic_light_id not in traffic_light_ids:
                    traffic_light_ids.append(traffic_light_id)
        return tuple(traffic_light_ids)

    def generate_trips(self) -> Path:
        """Generate trips and route files for the generated network."""
        if self.traffic_light_ids is None:
            self.traffic_light_ids = self.discover_traffic_light_ids()
            
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

        argv = self._random_trips_argv(
            trip_fringe_factor=self.trip_fringe_factor,
            allow_fringe=self.allow_fringe,
        )
        
        try:
            self._run_random_trips(argv)
        except subprocess.CalledProcessError:
            fallback_argv = self._random_trips_argv(
                trip_fringe_factor=None,
                allow_fringe=False,
            )
            print(
                "Fringe trip generation failed; falling back to random trips across all valid edges."
            )
            self._run_random_trips(fallback_argv)

        return self.routes_file

    def generate_peak_trips(self, peak_end: int = 1800, peak_period: float | None = None, offpeak_period: float | None = None) -> Path:
        """Generate realistic peak/off-peak demand pattern."""
        # Default to self.trip_period if not specified
        if peak_period is None:
            peak_period = self.trip_period
        if offpeak_period is None:
            offpeak_period = self.trip_period * 3.0
    
        # Discover traffic light IDs if not already initialised
        if self.traffic_light_ids is None:
            self.traffic_light_ids = self.discover_traffic_light_ids()
    
        peak_trips = self.output_dir / "peak.trips.xml"
        offpeak_trips = self.output_dir / "offpeak.trips.xml"
    
        base_args = [
            sys.executable, str(self.random_trips_file),
            "-n", str(self.network_file),
            "--fringe-factor", str(self.trip_fringe_factor or 1),
            "--allow-fringe",
            "--edge-permission", "passenger",
            "--speed-exponent", "2",
            "--random-depart",
            "--seed", str(self.seed),
        ]
    
        if self.min_distance is not None:
            base_args.extend(["--min-distance", str(self.min_distance)])
    
        if self.vtypes_file is not None:
            base_args.extend(["--additional-files", str(self.vtypes_file)])
    
        # Peak trips
        subprocess.run(base_args + [
            "-o", str(peak_trips),
            "--begin", "0",
            "--end", str(peak_end),
            "--period", str(peak_period),
            "--prefix", "peak_",
        ], check=True)
    
        # Off-peak trips
        subprocess.run(base_args + [
            "-o", str(offpeak_trips),
            "--begin", str(peak_end),
            "--end", str(self.trip_end),
            "--period", str(offpeak_period),
            "--prefix", "offpeak_",
        ], check=True)
    
        # Merge with duarouter
        subprocess.run([
            "duarouter",
            "-n", str(self.network_file),
            "--route-files", f"{peak_trips},{offpeak_trips}",
            "-o", str(self.routes_file),
            "--ignore-errors",
            "--seed", str(self.seed),
            "--no-step-log",
        ], check=True)
    
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
