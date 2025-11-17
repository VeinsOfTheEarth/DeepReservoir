# metadata.py

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any

def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "src").exists() and (p / "data").exists():
            return p
    for p in here.parents:
        if p.name == "src":
            return p.parent
    return here.parent

@dataclass
class Metadata:
    base_dir: Path = field(default_factory=repo_root)

    # explicit, no-magic configs: edit these dicts directly
    daily_series: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    continuous_series: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tables: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def _abs(self, p: Path | str) -> Path:
        p = p if isinstance(p, Path) else Path(p)
        return p if p.is_absolute() else (self.base_dir / p)

    def resolve_paths(self) -> "Metadata":
        for group in (self.daily_series, self.continuous_series, self.tables):
            for _, cfg in group.items():
                cfg["path"] = self._abs(cfg["path"])
        return self

# Rename of former "build_scaffold" → descriptive:
def project_metadata() -> Metadata:
    """
    Return the project's default metadata (paths, index, columns) with
    all relative paths resolved against the repo root.
    """
    m = Metadata()

    # DAILY time series (single-day index; all already in imperial)
    m.daily_series = {
        
        # Historic reservor data
        "reservoir": {
            "path": "data/navajo_reservoir_historic/Clipped_NAVAJORESERVOIR08-18-2024T16.48.23.csv",
            "reader": "csv",
            "index": {"name": "Date", "format": "%d-%b-%y"},
            "columns": {
                "Total Release (cfs)": "release_cfs",
                "Storage (af)": "storage_af",
                "Elevation (feet)": "elev_ft",
            },
            "required": ("release_cfs", "storage_af", "elev_ft"),
            "duplicates": "last",
        },
        "inflow": {
            "path": "data/navajo_reservoir_historic/Clipped_NAVAJORESERVOIR08-18-2024T16.48.23.csv",
            "reader": "csv",
            "index": {"name": "Date", "format": "%d-%b-%y"},
            "columns": {"Inflow** (cfs)": "inflow_cfs"},
            "required": ("inflow_cfs",),
            "duplicates": "last",
        },
        "evaporation": {
            "path": "data/navajo_reservoir_historic/Clipped_NAVAJORESERVOIR08-18-2024T16.48.23.csv",
            "reader": "csv",
            "index": {"name": "Date", "format": "%d-%b-%y"},
            "columns": {"Evaporation (af)": "evap_af"},
            "required": ("evap_af",),
            "duplicates": "last",
        },
        
        # Daily gages
        "sj_archuleta": {
            "path": "data/daily_flows/daily_sj_archuleta.csv",
            "reader": "csv",
            "index": {"name": "time", "format": None},
            "columns": {"value": "sj_archuleta_q_cfs"},
            "required": ("sj_archuleta_q_cfs",),
        },
        "sj_farmington": {
            "path": "data/daily_flows/daily_sj_farmington.csv",
            "reader": "csv",
            "index": {"name": "time", "format": None},
            "columns": {"value": "sj_farmington_q_cfs"},
            "required": ("sj_farmington_q_cfs",),
        },
        "sj_fourcorners": {
            "path": "data/daily_flows/daily_sj_fourcorners.csv",
            "reader": "csv",
            "index": {"name": "time", "format": None},
            "columns": {"value": "sj_fourcorners_q_cfs"},
            "required": ("sj_fourcorners_q_cfs",),
        },
        "sj_shiprock": {
            "path": "data/daily_flows/daily_sj_shiprock.csv",
            "reader": "csv",
            "index": {"name": "time", "format": None},
            "columns": {"value": "sj_shiprock_q_cfs"},
            "required": ("sj_shiprock_q_cfs",),
        },
        "sj_bluff": {
            "path": "data/daily_flows/daily_sj_bluff.csv",
            "reader": "csv",
            "index": {"name": "time", "format": None},
            "columns": {"value": "sj_bluff_q_cfs"},
            "required": ("sj_bluff_q_cfs",),
        },
        "animas_farmington": {
            "path": "data/daily_flows/daily_animas_farmington.csv",
            "reader": "csv",
            "index": {"name": "time", "format": None},
            "columns": {"value": "animas_farmington_q_cfs"},
            "required": ("animas_farmington_q_cfs",),
        },
        "chaco_waterflow": {
            "path": "data/daily_flows/daily_chaco_waterflow.csv",
            "reader": "csv",
            "index": {"name": "time", "format": None},
            "columns": {"value": "chaco_waterflow_q_cfs"},
            "required": ("chaco_waterflow_q_cfs",),
        },
        "chinle_mexicanwater": {
            "path": "data/daily_flows/daily_chinle_mexicanwater.csv",
            "reader": "csv",
            "index": {"name": "time", "format": None},
            "columns": {"value": "chinle_mexicanwater_q_cfs"},
            "required": ("chinle_mexicanwater_q_cfs",),
        },
        "laplata_farmington": {
            "path": "data/daily_flows/daily_laplata_farmington.csv",
            "reader": "csv",
            "index": {"name": "time", "format": None},
            "columns": {"value": "laplata_farmington_q_cfs"},
            "required": ("laplata_farmington_q_cfs",),
        },
        "mancos_towaoc": {
            "path": "data/daily_flows/daily_mancos_towaoc.csv",
            "reader": "csv",
            "index": {"name": "time", "format": None},
            "columns": {"value": "mancos_towaoc_q_cfs"},
            "required": ("mancos_towaoc_q_cfs",),
        },

        # NIIP
        "niip_historic": {
            "path": "data/niip/NAVAJOINDIANIRRIGATIONPROJECT07-17-2025T13.21.47.csv",
            "reader": "csv",
            "read_kwargs": {
                "usecols": ["Date", "Flow (cfs)"],  # keep only what we need
                "comment": "*",                      # drop footnote lines
                "skipinitialspace": True,            # harmless; trims after commas
            },
            "index": {"name": "Date", "format": "%d-%b-%y"},
            "columns": {"Flow (cfs)": "niip_flow_cfs"},
            "required": ("niip_flow_cfs",),
            "duplicates": "last",
    }
    }

    # CONTINUOUS/irregular time series (i.e. not daily)
    m.continuous_series = {
        "sj_archuleta": {
            "path": r"X:/Research/DeepReservoir/data/usgs_realtime_gages/sj_at_archuleta.txt",
            "reader": "usgs_irregular",
            "index": {"name": "datetime", "format": None},  
            "columns": {"value": "sj_archuleta_q_cfs"},
            "required": ("sj_archuleta_q_cfs",),
        },
        "sj_farmington": {
            "path": r"X:/Research/DeepReservoir/data/usgs_realtime_gages/sj_at_farmington.txt",
            "reader": "usgs_irregular",
            "index": {"name": "datetime", "format": None},
            "columns": {"value": "sj_farmington_q_cfs"},
            "required": ("sj_farmington_q_cfs",),
        },
        "sj_bluff": {
            "path": r"X:/Research/DeepReservoir/data/usgs_realtime_gages/sj_at_bluff.txt",
            "reader": "usgs_irregular",
            "index": {"name": "datetime", "format": None},
            "columns": {"value": "sj_bluff_q_cfs"},
            "required": ("sj_bluff_q_cfs",),
        },
        "sj_fourcorners": {
            "path": r"X:/Research/DeepReservoir/data/usgs_realtime_gages/sj_at_fourcorners.txt",
            "reader": "usgs_irregular",
            "index": {"name": "datetime", "format": None},
            "columns": {"value": "sj_fourcorners_q_cfs"},
            "required": ("sj_fourcorners_q_cfs",),
        },
        "sj_shiprock": {
            "path": r"X:/Research/DeepReservoir/data/usgs_realtime_gages/sj_at_shiprock.txt",
            "reader": "usgs_irregular",
            "index": {"name": "datetime", "format": None},
            "columns": {"value": "sj_shiprock_q_cfs"},
            "required": ("sj_shiprock_q_cfs",),
        },
        "animas_farmington": {
            "path": r"X:/Research/DeepReservoir/data/usgs_realtime_gages/animas_at_farmington.txt",
            "reader": "usgs_irregular",
            "index": {"name": "datetime", "format": None},
            "columns": {"value": "animas_farmington_q_cfs"},
            "required": ("animas_farmington_q_cfs",),
        },

        # SWE 
        "swe_upper_sj": {
            "path": r"X:\Research\DeepReservoir\data\swe\UpperSJ.parquet",
            "reader": "parquet",
            "index": {"name": "date", "format": "%Y-%m-%d %H:%M"},
            "columns": {"snow_depth_water_equivalent": "swe_m"},
            "required": ("swe_m",),
        },
        "swe_animas": {
            "path": r"X:\Research\DeepReservoir\data\swe\Animas.parquet",
            "reader": "parquet",
            "index": {"name": "date", "format": "%Y-%m-%d %H:%M"},
            "columns": {"snow_depth_water_equivalent": "swe_m"},
            "required": ("swe_m",),
        },
        "swe_lospinos": {
            "path": r"X:\Research\DeepReservoir\data\swe\LosPinos.parquet",
            "reader": "parquet",
            "index": {"name": "date", "format": "%Y-%m-%d %H:%M"},
            "columns": {"snow_depth_water_equivalent": "swe_m"},
            "required": ("swe_m",),
        },
        "swe_piedra": {
            "path": r"X:\Research\DeepReservoir\data\swe\Piedra.parquet",
            "reader": "parquet",
            "index": {"name": "date", "format": "%Y-%m-%d %H:%M"},
            "columns": {"snow_depth_water_equivalent": "swe_m"},
            "required": ("swe_m",),
        },
        "swe_spring": {
            "path": r"X:\Research\DeepReservoir\data\swe\Spring.parquet",
            "reader": "parquet",
            "index": {"name": "date", "format": "%Y-%m-%d %H:%M"},
            "columns": {"snow_depth_water_equivalent": "swe_m"},
            "required": ("swe_m",),
        },

    }

    # Non-time tables (E–A–S relationship, ...)
    m.tables = {
        "eas": {
            "path": "data/elevation_area_storage_relationships/elevation_storage_area_2019.csv",
            "reader": "csv",
            "index": {"name": None, "format": None},  # do not parse a datetime index
            "columns": {
                "Elevation (ft)": "elev_ft",
                "Area (acres)": "area_ac",
                "Capacity (af)": "capacity_af",
            },
            "required": ("elev_ft", "area_ac", "capacity_af"),
        },
    }

    return m.resolve_paths()
