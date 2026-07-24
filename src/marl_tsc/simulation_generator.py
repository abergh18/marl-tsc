"""Generate a small SUMO grid for MARL traffic-signal-control experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import runpy
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
        grid_number: int = 4,
        lane_number: int = 1,
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
        """Create a SUMO grid network, traffic demand, routes, and config."""

        self.output_dir = Path(output_dir)
        self.network_filename = network_filename
        self.trips_filename = trips_filename
        self.routes_filename = routes_filename
        self.config_filename = config_filename

        self.grid_number = grid_number
        self.lane_number = lane_number
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
        )

    def run_command(self, command: list[str]) -> None:
        """Run an external SUMO command and fail loudly on errors."""
        subprocess.run(command, check=True)

    def generate_network(self) -> Path:
        """Generate the SUMO network XML file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
        if self.traffic_light_ids is None:
            self.traffic_light_ids = self.discover_traffic_light_ids()
    
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
        if self.traffic_light_ids is None:
            self.traffic_light_ids = self.discover_traffic_light_ids()
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

    def generate_peak_trips(self, peak_end: int = 1800, peak_period: float = 1.0, offpeak_period: float = 3.0) -> Path:
        """Generate realistic peak/off-peak demand pattern."""
    
        peak_trips = self.output_dir / "peak.trips.xml"
        offpeak_trips = self.output_dir / "offpeak.trips.xml"
    
        # Save originals
        orig_begin = self.trip_begin
        orig_end = self.trip_end
        orig_period = self.trip_period
        orig_trips = self.trips_filename
    
        # Generate peak trips
        self.trips_filename = "peak.trips.xml"
        self.trip_begin = 0
        self.trip_end = peak_end
        self.trip_period = peak_period
        argv = self._random_trips_argv(trip_fringe_factor=self.trip_fringe_factor, allow_fringe=self.allow_fringe)
        argv.extend(["--prefix", "peak_"])
        self._run_random_trips(argv)
            
        argv = self._random_trips_argv(
            trip_fringe_factor=self.trip_fringe_factor,
            allow_fringe=self.allow_fringe,
        )
        self._run_random_trips(argv)
    
        # Generate off-peak trips
        self.trips_filename = "offpeak.trips.xml"
        self.trip_begin = peak_end
        self.trip_end = orig_end
        self.trip_period = offpeak_period
        argv = self._random_trips_argv(trip_fringe_factor=self.trip_fringe_factor, allow_fringe=self.allow_fringe)
        argv.extend(["--prefix", "offpeak_"])
        self._run_random_trips(argv)
            
        old_vtypes = self.vtypes_file
        self.vtypes_file = None
        
        argv = self._random_trips_argv(
            trip_fringe_factor=self.trip_fringe_factor,
            allow_fringe=self.allow_fringe,
        )
        self._run_random_trips(argv)
        
        self.vtypes_file = old_vtypes
    
        # Restore originals
        self.trip_begin = orig_begin
        self.trip_end = orig_end
        self.trip_period = orig_period
        self.trips_filename = orig_trips
    
        # Merge using duarouter
        subprocess.run([
            "duarouter",
            "-n", str(self.network_file),
            "--route-files", f"{peak_trips},{offpeak_trips}",
            "-o", str(self.routes_file),
            "--ignore-errors",
            "--seed", str(self.seed),
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
