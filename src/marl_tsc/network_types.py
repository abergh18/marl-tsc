"""Network types used by the SUMO simulation generator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
from string import ascii_uppercase
from urllib.parse import urlencode
from urllib.request import Request, urlopen


RunCommand = Callable[[list[str]], None]


class NetworkType(ABC):
    """Base class for objects that can create a SUMO network file."""

    @abstractmethod
    def generate(
        self,
        output_dir: Path,
        network_file: Path,
        run_command: RunCommand,
    ) -> None:
        """Create network_file inside output_dir."""


@dataclass
class GridNetwork(NetworkType):
    """Create a grid network."""

    grid_number: int = 4
    lane_number: int = 2

    def _traffic_light_ids(self) -> tuple[str, ...]:
        if self.grid_number < 3:
            raise ValueError("grid_number must be at least 3 to have internal traffic lights.")
        if self.grid_number > len(ascii_uppercase):
            raise ValueError("grid_number is too large for the simple A-Z grid naming scheme.")

        return tuple(
            f"{ascii_uppercase[column]}{row}"
            for column in range(1, self.grid_number - 1)
            for row in range(1, self.grid_number - 1)
        )

    def generate(
        self,
        output_dir: Path,
        network_file: Path,
        run_command: RunCommand,
    ) -> None:
        run_command(
            [
                "netgenerate",
                "--grid",
                f"--grid.number={self.grid_number}",
                f"--default.lanenumber={self.lane_number}",
                "--tls.layout",
                "incoming",
                "--tls.set",
                ",".join(self._traffic_light_ids()),
                "-o",
                str(network_file),
            ]
        )


@dataclass
class CityNetwork(NetworkType):
    """A small UK city network downloaded from OpenStreetMap."""

    city_name: str = "Bristol, UK"
    radius: int = 150
    lane_number: int = 2
    left_hand: bool = True

    def generate(
        self,
        output_dir: Path,
        network_file: Path,
        run_command: RunCommand,
    ) -> None:
        latitude, longitude = self._geocode_city()
        osm_file = output_dir / "city.osm.xml"
        self._download_osm(latitude, longitude, osm_file)

        command = [
            "netconvert",
            "--osm-files",
            str(osm_file),
            "--output-file",
            str(network_file),
            "--junctions.join",
            "--edges.join",
            "--geometry.remove",
            "--geometry.min-dist",
            "5",
            "--roundabouts.guess",
            "--no-turnarounds",
            "--tls.guess",
            "--tls.join",
            "--default.lanenumber",
            str(self.lane_number),
            "--default.lanewidth",
            "3.2",
        ]
        if self.left_hand:
            command.append("--lefthand")

        run_command(command)

    def _geocode_city(self) -> tuple[float, float]:
        query = urlencode({"format": "json", "limit": "1", "q": self.city_name})
        request = Request(
            f"https://nominatim.openstreetmap.org/search?{query}",
            headers={"User-Agent": "marl-tsc-class-project/1.0"},
        )
        with urlopen(request, timeout=30) as response:
            results = json.loads(response.read().decode("utf-8"))

        if not results:
            raise ValueError(f"Could not find coordinates for {self.city_name!r}.")

        return float(results[0]["lat"]), float(results[0]["lon"])

    def _download_osm(self, latitude: float, longitude: float, osm_file: Path) -> None:
        highway_types = "motorway|trunk|primary|secondary|tertiary|unclassified|residential"
        overpass_query = f"""[out:xml][timeout:60];
way["highway"~"{highway_types}"](around:{self.radius},{latitude},{longitude});
(._;>;);
out body;"""
        body = urlencode({"data": overpass_query}).encode("utf-8")
        request = Request(
            "https://overpass-api.de/api/interpreter",
            data=body,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "marl-tsc-class-project/1.0",
            },
        )
        with urlopen(request, timeout=90) as response:
            osm_file.write_bytes(response.read())
