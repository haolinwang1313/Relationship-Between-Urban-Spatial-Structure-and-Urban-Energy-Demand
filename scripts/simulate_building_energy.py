#!/usr/bin/env python3
"""Generate simplified EnergyPlus models per building and aggregate energy outputs.

The script uses default prototype assumptions (config/energy_prototypes.json) to
transform each building footprint into a single-zone EnergyPlus model powered by
HVACTemplate:Zone:IdealLoadsAirSystem. It then runs the EnergyPlus CLI, extracts
cooling/heating loads plus lights/equipment electricity from the SQLite output,
and converts them into annual/monthly kWh tables keyed by building_id.

Usage example
-------------
source .venv_geo/bin/activate && \
python scripts/simulate_building_energy.py \
    --limit 5 \
    --weather-file Eplus/EnergyPlus-25.1.0-68a4a7c774-Linux-Ubuntu22.04-x86_64/WeatherData/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from geomeppy import IDF

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "Data" / "Processed"
BUILDING_PATH = PROCESSED_DIR / "Building" / "xinwu_buildings.gpkg"
LAND_USE_PATH = PROCESSED_DIR / "Land_Use" / "xinwu_land_use.gpkg"
CONFIG_PATH = PROJECT_ROOT / "config" / "energy_prototypes.json"
ENERGYPLUS_ROOT = PROJECT_ROOT / "Eplus" / "EnergyPlus-24.1.0-9d7789a3ac-Linux-Ubuntu22.04-x86_64"
DEFAULT_WEATHER = ENERGYPLUS_ROOT / "WeatherData" / "CHN_JS_Wuxi_Proxy_583580.epw"
ENERGY_DIR = PROCESSED_DIR / "Energy"
RUNS_DIR = ENERGY_DIR / "runs"
DEFAULT_OUTPUT = ENERGY_DIR / "xinwu_building_energy_sample.csv"
BASE_IDF_PATH = PROJECT_ROOT / "config" / "base_energy_model.idf"
JOULE_TO_KWH = 1.0 / 3_600_000.0
MIN_AREA_M2 = 25.0
MIN_HEIGHT_M = 3.3
AVG_FLOOR_HEIGHT = 3.3

os.environ.setdefault("EPLUS_IDD", str(ENERGYPLUS_ROOT / "Energy+.idd"))
IDF.setiddname(os.environ["EPLUS_IDD"])


@dataclass
class Prototype:
    name: str
    schedule_key: str
    wwr: float
    wall_u: float
    roof_u: float
    window_u: float
    people_per_m2: float
    lights_w_per_m2: float
    equipment_w_per_m2: float
    infiltration_ach: float
    cooling_cop: float
    heating_cop: float

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, float]) -> "Prototype":
        required = [
            "schedule_key",
            "wwr",
            "wall_u",
            "roof_u",
            "window_u",
            "people_per_m2",
            "lights_w_per_m2",
            "equipment_w_per_m2",
            "infiltration_ach",
            "cooling_cop",
            "heating_cop",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Prototype {name} missing keys: {missing}")
        return cls(
            name=name,
            schedule_key=data["schedule_key"],
            wwr=float(data["wwr"]),
            wall_u=float(data["wall_u"]),
            roof_u=float(data["roof_u"]),
            window_u=float(data["window_u"]),
            people_per_m2=float(data["people_per_m2"]),
            lights_w_per_m2=float(data["lights_w_per_m2"]),
            equipment_w_per_m2=float(data["equipment_w_per_m2"]),
            infiltration_ach=float(data["infiltration_ach"]),
            cooling_cop=max(0.1, float(data["cooling_cop"])),
            heating_cop=max(0.1, float(data["heating_cop"]))
        )


SCHEDULE_TEMPLATES: Dict[str, Dict[str, str]] = {
    "residential": {
        "occupancy": """
Schedule:Compact,
  {name},
  Fraction,
  Through: 12/31,
  For: Weekdays,
  Until: 06:00,0.2,
  Until: 09:00,0.7,
  Until: 18:00,0.5,
  Until: 22:00,0.9,
  Until: 24:00,0.4,
  For: Weekends Holidays,
  Until: 09:00,0.5,
  Until: 18:00,0.8,
  Until: 24:00,0.6;
""",
        "lights": """
Schedule:Compact,
  {name},
  Fraction,
  Through: 12/31,
  For: Weekdays,
  Until: 06:00,0.1,
  Until: 09:00,0.4,
  Until: 18:00,0.3,
  Until: 22:00,0.8,
  Until: 24:00,0.4,
  For: Weekends Holidays,
  Until: 09:00,0.4,
  Until: 18:00,0.6,
  Until: 24:00,0.5;
""",
        "equipment": """
Schedule:Compact,
  {name},
  Fraction,
  Through: 12/31,
  For: Weekdays,
  Until: 06:00,0.2,
  Until: 09:00,0.6,
  Until: 18:00,0.5,
  Until: 23:00,0.9,
  Until: 24:00,0.4,
  For: Weekends Holidays,
  Until: 10:00,0.4,
  Until: 18:00,0.7,
  Until: 24:00,0.5;
""",
    },
    "office": {
        "occupancy": """
Schedule:Compact,
  {name},
  Fraction,
  Through: 12/31,
  For: Weekdays,
  Until: 06:00,0.05,
  Until: 08:00,0.2,
  Until: 12:00,0.9,
  Until: 17:00,0.95,
  Until: 19:00,0.2,
  Until: 24:00,0.05,
  For: Weekends Holidays,
  Until: 24:00,0.05;
""",
        "lights": """
Schedule:Compact,
  {name},
  Fraction,
  Through: 12/31,
  For: Weekdays,
  Until: 06:00,0.1,
  Until: 08:00,0.5,
  Until: 18:00,1.0,
  Until: 20:00,0.3,
  Until: 24:00,0.1,
  For: Weekends Holidays,
  Until: 24:00,0.1;
""",
        "equipment": """
Schedule:Compact,
  {name},
  Fraction,
  Through: 12/31,
  For: Weekdays,
  Until: 06:00,0.2,
  Until: 08:00,0.5,
  Until: 18:00,1.0,
  Until: 20:00,0.4,
  Until: 24:00,0.2,
  For: Weekends Holidays,
  Until: 24:00,0.2;
""",
    },
    "retail": {
        "occupancy": """
Schedule:Compact,
  {name},
  Fraction,
  Through: 12/31,
  For: Weekdays Weekends,
  Until: 08:00,0.05,
  Until: 10:00,0.5,
  Until: 21:00,0.95,
  Until: 23:00,0.3,
  Until: 24:00,0.05,
  For: Holidays,
  Until: 24:00,0.1;
""",
        "lights": """
Schedule:Compact,
  {name},
  Fraction,
  Through: 12/31,
  For: Weekdays Weekends,
  Until: 08:00,0.2,
  Until: 10:00,0.8,
  Until: 21:00,1.0,
  Until: 23:00,0.4,
  Until: 24:00,0.2,
  For: Holidays,
  Until: 24:00,0.2;
""",
        "equipment": """
Schedule:Compact,
  {name},
  Fraction,
  Through: 12/31,
  For: Weekdays Weekends,
  Until: 08:00,0.3,
  Until: 10:00,0.6,
  Until: 21:00,0.9,
  Until: 23:00,0.5,
  Until: 24:00,0.3,
  For: Holidays,
  Until: 24:00,0.3;
""",
    },
    "education": {
        "occupancy": """
Schedule:Compact,
  {name},
  Fraction,
  Through: 12/31,
  For: Weekdays,
  Until: 07:00,0.05,
  Until: 09:00,0.8,
  Until: 15:00,1.0,
  Until: 18:00,0.3,
  Until: 24:00,0.05,
  For: Weekends Holidays,
  Until: 24:00,0.05;
""",
        "lights": """
Schedule:Compact,
  {name},
  Fraction,
  Through: 12/31,
  For: Weekdays,
  Until: 07:00,0.1,
  Until: 09:00,0.8,
  Until: 17:00,1.0,
  Until: 19:00,0.4,
  Until: 24:00,0.1,
  For: Weekends Holidays,
  Until: 24:00,0.1;
""",
        "equipment": """
Schedule:Compact,
  {name},
  Fraction,
  Through: 12/31,
  For: Weekdays,
  Until: 07:00,0.2,
  Until: 09:00,0.7,
  Until: 17:00,0.9,
  Until: 19:00,0.3,
  Until: 24:00,0.2,
  For: Weekends Holidays,
  Until: 24:00,0.2;
""",
    },
    "industrial": {
        "occupancy": """
Schedule:Compact,
  {name},
  Fraction,
  Through: 12/31,
  For: AllDays,
  Until: 24:00,0.3;
""",
        "lights": """
Schedule:Compact,
  {name},
  Fraction,
  Through: 12/31,
  For: AllDays,
  Until: 24:00,0.8;
""",
        "equipment": """
Schedule:Compact,
  {name},
  Fraction,
  Through: 12/31,
  For: AllDays,
  Until: 24:00,1.0;
""",
    },
}

SCHEDULE_LIMITS = """
ScheduleTypeLimits,
  Fraction,
  0.0,
  1.0,
  CONTINUOUS;

ScheduleTypeLimits,
  Temperature,
  -60,
  80,
  CONTINUOUS;

ScheduleTypeLimits,
  Any Number;
"""

ALWAYS_ON = """
Schedule:Compact,
  {name},
  Fraction,
  Through: 12/31,
  For: AllDays,
  Until: 24:00,1.0;
"""
#渗透，整年的透风量不变
ACTIVITY_SCHEDULE = """
Schedule:Compact,
  {name},
  Any Number,
  Through: 12/31,
  For: AllDays,
  Until: 24:00,120.0;
"""

CLOTHING_SCHEDULE = """
Schedule:Compact,
  {name},
  Any Number,
  Through: 12/31,
  For: AllDays,
  Until: 24:00,1.0;
"""

CONTROL_TYPE_SCHEDULE = """
Schedule:Compact,
  {name},
  Control Type,
  Through: 12/31,
  For: AllDays,
  Until: 24:00,4.0;
"""


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in value)


def make_labels(building_id: str) -> Tuple[str, str]:
    building_label = sanitize_name(f"B_{building_id}")
    return building_label, f"ZONE_{building_label}"


def load_prototype_config(path: Path) -> Tuple[Dict[str, Prototype], Dict[str, str], Dict[str, str], str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    prototypes = {name: Prototype.from_dict(name, info) for name, info in data["prototypes"].items()}
    type_map = data.get("type_mapping", {})
    land_map = data.get("land_class_mapping", {})
    default_proto = data.get("default_prototype")
    if default_proto not in prototypes:
        raise ValueError("default_prototype must exist in prototypes dictionary")
    return prototypes, type_map, land_map, default_proto


def assign_land_use(buildings: gpd.GeoDataFrame, land_use_path: Path) -> gpd.GeoDataFrame:
    if "land_class" in buildings.columns:
        return buildings
    land_use = gpd.read_file(land_use_path)
    if land_use.crs != buildings.crs:
        land_use = land_use.to_crs(buildings.crs)
    centroids = buildings.copy()
    centroids["geometry"] = centroids.geometry.centroid
    joined = gpd.sjoin(centroids, land_use[["Class", "geometry"]], how="left", predicate="within").reset_index()
    joined = joined.rename(columns={"index": "building_index"})
    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])
    land_series = joined.groupby("building_index")["Class"].first()
    buildings = buildings.copy()
    buildings["land_class"] = buildings.index.to_series().map(land_series)
    return buildings


def determine_prototype(row, type_map: Dict[str, str], land_map: Dict[str, str], default_proto: str) -> str:
    proto = None
    if "type" in row and pd.notna(row["type"]):
        key = str(int(round(row["type"])))
        proto = type_map.get(key)
    if proto is None and "land_class" in row and pd.notna(row["land_class"]):
        proto = land_map.get(str(int(round(row["land_class"]))))
    return proto or default_proto


def compute_geometry(area_m2: float, height_m: float) -> Tuple[float, float, float, int]:
    area = max(MIN_AREA_M2, float(area_m2))
    height = max(MIN_HEIGHT_M, float(height_m) if height_m and not math.isnan(height_m) else MIN_HEIGHT_M)
    floors = max(1, int(round(height / AVG_FLOOR_HEIGHT)))
    zone_height = floors * AVG_FLOOR_HEIGHT
    width = max(5.0, math.sqrt(area))
    length = area / width
    return length, width, zone_height, floors


def format_vertices(vertices: List[Tuple[float, float, float]]) -> str:
    lines = []
    for idx, (x, y, z) in enumerate(vertices):
        suffix = "," if idx < len(vertices) - 1 else ";"
        lines.append(f"  {x:.3f},{y:.3f},{z:.3f}{suffix}")
    return "\n".join(lines)


def add_schedule_from_template(idf: IDF, template: str, schedule_name: str) -> str:
    lines = [line.strip() for line in template.format(name=schedule_name).splitlines() if line.strip()]
    if len(lines) < 4:
        raise ValueError(f"Invalid schedule template for {schedule_name}")
    schedule_type = lines[2].rstrip(",").strip()
    body = [line.rstrip(",;").strip() for line in lines[3:]]
    schedule = idf.newidfobject("SCHEDULE:COMPACT")
    schedule.Name = schedule_name
    schedule.Schedule_Type_Limits_Name = schedule_type
    for idx, entry in enumerate(body, start=1):
        setattr(schedule, f"Field_{idx}", entry)
    return schedule_name


def ensure_constructions(idf: IDF, proto: Prototype) -> Dict[str, str]:
    suffix = proto.name.upper()
    wall = idf.newidfobject(
        "MATERIAL:NOMASS",
        Name=f"MAT_WALL_{suffix}",
        Roughness="Smooth",
        Thermal_Resistance=1.0 / max(proto.wall_u, 0.1),
        Thermal_Absorptance=0.9,
        Solar_Absorptance=0.7,
        Visible_Absorptance=0.7,
    )
    roof = idf.newidfobject(
        "MATERIAL:NOMASS",
        Name=f"MAT_ROOF_{suffix}",
        Roughness="Smooth",
        Thermal_Resistance=1.0 / max(proto.roof_u, 0.1),
        Thermal_Absorptance=0.9,
        Solar_Absorptance=0.7,
        Visible_Absorptance=0.7,
    )
    floor = idf.newidfobject(
        "MATERIAL:NOMASS",
        Name=f"MAT_FLOOR_{suffix}",
        Roughness="Smooth",
        Thermal_Resistance=2.0,
        Thermal_Absorptance=0.9,
        Solar_Absorptance=0.7,
        Visible_Absorptance=0.7,
    )
    glazing = idf.newidfobject(
        "WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM",
        Name=f"MAT_WIN_{suffix}",
        UFactor=proto.window_u,
        Solar_Heat_Gain_Coefficient=0.5,
        Visible_Transmittance=0.6,
    )
    wall_cons = idf.newidfobject(
        "CONSTRUCTION",
        Name=f"CONS_WALL_{suffix}",
        Outside_Layer=wall.Name,
    )
    roof_cons = idf.newidfobject(
        "CONSTRUCTION",
        Name=f"CONS_ROOF_{suffix}",
        Outside_Layer=roof.Name,
    )
    floor_cons = idf.newidfobject(
        "CONSTRUCTION",
        Name=f"CONS_FLOOR_{suffix}",
        Outside_Layer=floor.Name,
    )
    win_cons = idf.newidfobject(
        "CONSTRUCTION",
        Name=f"CONS_WIN_{suffix}",
        Outside_Layer=glazing.Name,
    )
    return {
        "wall": wall_cons.Name,
        "roof": roof_cons.Name,
        "floor": floor_cons.Name,
        "window": win_cons.Name,
    }


def add_geometry_block(
    idf: IDF,
    zone_name: str,
    length: float,
    width: float,
    height: float,
    constructions: Dict[str, str],
    wwr: float,
) -> float:
    surface_start = len(idf.idfobjects["BUILDINGSURFACE:DETAILED"])
    idf.add_block(name=zone_name, coordinates=[(0.0, 0.0), (length, 0.0), (length, width), (0.0, width)], height=height, num_stories=1)
    zone = idf.idfobjects["ZONE"][-1]
    zone.Name = zone_name
    zone.Ceiling_Height = height
    volume = length * width * height
    zone.Volume = volume
    new_surfaces = idf.idfobjects["BUILDINGSURFACE:DETAILED"][surface_start:]
    for surf in new_surfaces:
        surf.Zone_Name = zone_name
        surf.Space_Name = ""
        stype = surf.Surface_Type.lower()
        if stype == "floor":
            surf.Construction_Name = constructions["floor"]
            surf.Outside_Boundary_Condition = "Ground"
            surf.Sun_Exposure = "NoSun"
            surf.Wind_Exposure = "NoWind"
            surf.View_Factor_to_Ground = 1.0
        elif stype == "roof":
            surf.Construction_Name = constructions["roof"]
            surf.Outside_Boundary_Condition = "Outdoors"
            surf.Sun_Exposure = "SunExposed"
            surf.Wind_Exposure = "WindExposed"
            surf.View_Factor_to_Ground = 0.0
        else:
            surf.Construction_Name = constructions["wall"]
            surf.Outside_Boundary_Condition = "Outdoors"
            surf.Sun_Exposure = surf.Sun_Exposure or "SunExposed"
            surf.Wind_Exposure = surf.Wind_Exposure or "WindExposed"
            surf.View_Factor_to_Ground = 0.5
    if wwr > 0:
        idf.set_wwr(max(0.0, min(0.9, wwr)), construction=constructions["window"], force=True)
    return volume


def add_internal_loads(
    idf: IDF,
    zone_name: str,
    proto: Prototype,
    volume_m3: float,
    infiltration_schedule: str,
    occ_schedule: str,
    light_schedule: str,
    equip_schedule: str,
    activity_schedule: str,
    clothing_schedule: str,
) -> None:
    infiltration_flow = proto.infiltration_ach * volume_m3 / 3600.0
    idf.newidfobject(
        "ZONEINFILTRATION:DESIGNFLOWRATE",
        Name=f"INFIL_{zone_name}",
        Zone_or_ZoneList_or_Space_or_SpaceList_Name=zone_name,
        Schedule_Name=infiltration_schedule,
        Design_Flow_Rate_Calculation_Method="Flow/Zone",
        Design_Flow_Rate=infiltration_flow,
        Constant_Term_Coefficient=1.0,
        Temperature_Term_Coefficient=0.0,
        Velocity_Term_Coefficient=0.0,
        Velocity_Squared_Term_Coefficient=0.0,
    )
    idf.newidfobject(
        "PEOPLE",
        Name=f"PPL_{zone_name}",
        Zone_or_ZoneList_or_Space_or_SpaceList_Name=zone_name,
        Number_of_People_Schedule_Name=occ_schedule,
        Number_of_People_Calculation_Method="People/Area",
        People_per_Floor_Area=proto.people_per_m2,
        Fraction_Radiant=0.3,
        Activity_Level_Schedule_Name=activity_schedule,
        Clothing_Insulation_Calculation_Method="ClothingInsulationSchedule",
        Clothing_Insulation_Schedule_Name=clothing_schedule,
    )
    idf.newidfobject(
        "LIGHTS",
        Name=f"LGT_{zone_name}",
        Zone_or_ZoneList_or_Space_or_SpaceList_Name=zone_name,
        Schedule_Name=light_schedule,
        Design_Level_Calculation_Method="Watts/Area",
        Watts_per_Floor_Area=proto.lights_w_per_m2,
        Return_Air_Fraction=0.0,
        Fraction_Radiant=0.3,
        Fraction_Visible=0.2,
        Fraction_Replaceable=0.0,
    )
    idf.newidfobject(
        "ELECTRICEQUIPMENT",
        Name=f"EQ_{zone_name}",
        Zone_or_ZoneList_or_Space_or_SpaceList_Name=zone_name,
        Schedule_Name=equip_schedule,
        Design_Level_Calculation_Method="Watts/Area",
        Watts_per_Floor_Area=proto.equipment_w_per_m2,
        Fraction_Latent=0.0,
        Fraction_Radiant=0.3,
        Fraction_Lost=0.3,
    )


def add_zone_hvac(
    idf: IDF,
    zone_name: str,
    heat_schedule: str,
    cool_schedule: str,
    control_schedule: str,
) -> None:
    supply_node = f"{zone_name}_Supply"
    exhaust_node = f"{zone_name}_Exhaust"
    air_node = f"{zone_name}_Air"
    return_node = f"{zone_name}_Return"
    equip_list = f"EQUIPLIST_{zone_name}"
    ideal_name = f"IDEAL_{zone_name}"
    dsoa_name = f"DSOA_{zone_name}"
    idf.newidfobject(
        "DESIGNSPECIFICATION:OUTDOORAIR",
        Name=dsoa_name,
        Outdoor_Air_Method="Flow/Person",
        Outdoor_Air_Flow_per_Person=0.00944,
    )
    idf.newidfobject(
        "ZONECONTROL:THERMOSTAT",
        Name=f"CTRL_{zone_name}",
        Zone_or_ZoneList_Name=zone_name,
        Control_Type_Schedule_Name=control_schedule,
        Control_1_Object_Type="ThermostatSetpoint:DualSetpoint",
        Control_1_Name=f"TSTAT_{zone_name}",
    )
    idf.newidfobject(
        "THERMOSTATSETPOINT:DUALSETPOINT",
        Name=f"TSTAT_{zone_name}",
        Heating_Setpoint_Temperature_Schedule_Name=heat_schedule,
        Cooling_Setpoint_Temperature_Schedule_Name=cool_schedule,
    )
    equip = idf.newidfobject(
        "ZONEHVAC:EQUIPMENTLIST",
        Name=equip_list,
        Load_Distribution_Scheme="SequentialLoad",
    )
    equip.Zone_Equipment_1_Object_Type = "ZoneHVAC:IdealLoadsAirSystem"
    equip.Zone_Equipment_1_Name = ideal_name
    equip.Zone_Equipment_1_Cooling_Sequence = 1
    equip.Zone_Equipment_1_Heating_or_NoLoad_Sequence = 1
    idf.newidfobject(
        "ZONEHVAC:EQUIPMENTCONNECTIONS",
        Zone_Name=zone_name,
        Zone_Conditioning_Equipment_List_Name=equip_list,
        Zone_Air_Inlet_Node_or_NodeList_Name=supply_node,
        Zone_Air_Exhaust_Node_or_NodeList_Name=exhaust_node,
        Zone_Air_Node_Name=air_node,
        Zone_Return_Air_Node_or_NodeList_Name=return_node,
    )
    idf.newidfobject(
        "ZONEHVAC:IDEALLOADSAIRSYSTEM",
        Name=ideal_name,
        Availability_Schedule_Name="AlwaysOn",
        Zone_Supply_Air_Node_Name=supply_node,
        Zone_Exhaust_Air_Node_Name=exhaust_node,
        Heating_Availability_Schedule_Name="AlwaysOn",
        Cooling_Availability_Schedule_Name="AlwaysOn",
        Heating_Limit="NoLimit",
        Cooling_Limit="NoLimit",
        Design_Specification_Outdoor_Air_Object_Name=dsoa_name,
        Dehumidification_Control_Type="ConstantSensibleHeatRatio",
        Cooling_Sensible_Heat_Ratio=0.7,
        Humidification_Control_Type="None",
        Outdoor_Air_Economizer_Type="NoEconomizer",
        Heat_Recovery_Type="None",
        Sensible_Heat_Recovery_Effectiveness=0.7,
        Latent_Heat_Recovery_Effectiveness=0.65,
    )


def add_output_requests(idf: IDF, zone_name: str) -> None:
    for variable in [
        "Zone Ideal Loads Supply Air Total Cooling Energy",
        "Zone Ideal Loads Supply Air Total Heating Energy",
        "Zone Lights Electricity Energy",
        "Zone Electric Equipment Electricity Energy",
    ]:
        idf.newidfobject(
            "OUTPUT:VARIABLE",
            Key_Value="*",
            Variable_Name=variable,
            Reporting_Frequency="Hourly",
        )


def build_idf(
    building_id: str,
    building_label: str,
    zone_name: str,
    area_m2: float,
    height_m: float,
    proto: Prototype,
) -> str:
    length, width, zone_height, _ = compute_geometry(area_m2, height_m)
    idf = IDF(str(BASE_IDF_PATH))
    idf.idfobjects["BUILDING"][0].Name = building_label
    constructions = ensure_constructions(idf, proto)

    occ_schedule = add_schedule_from_template(
        idf, SCHEDULE_TEMPLATES[proto.schedule_key]["occupancy"], f"SCH_OCC_{building_label}"
    )
    light_schedule = add_schedule_from_template(
        idf, SCHEDULE_TEMPLATES[proto.schedule_key]["lights"], f"SCH_LGT_{building_label}"
    )
    equip_schedule = add_schedule_from_template(
        idf, SCHEDULE_TEMPLATES[proto.schedule_key]["equipment"], f"SCH_EQ_{building_label}"
    )
    activity_schedule = add_schedule_from_template(idf, ACTIVITY_SCHEDULE, f"SCH_ACTIVITY_{building_label}")
    heat_schedule = add_schedule_from_template(
        idf,
        """
Schedule:Compact,
  {name},
  Temperature,
  Through: 12/31,
  For: AllDays,
  Until: 24:00,20.0;
""",
        f"SCH_HEAT_{building_label}",
    )
    cool_schedule = add_schedule_from_template(
        idf,
        """
Schedule:Compact,
  {name},
  Temperature,
  Through: 12/31,
  For: AllDays,
  Until: 24:00,26.0;
""",
        f"SCH_COOL_{building_label}",
    )
    infiltration_schedule = add_schedule_from_template(idf, ALWAYS_ON, f"SCH_INFIL_{building_label}")
    clothing_schedule = add_schedule_from_template(idf, CLOTHING_SCHEDULE, f"SCH_CLO_{building_label}")
    control_schedule = add_schedule_from_template(idf, CONTROL_TYPE_SCHEDULE, f"SCH_CTRL_{building_label}")

    volume = add_geometry_block(idf, zone_name, length, width, zone_height, constructions, proto.wwr)
    add_internal_loads(
        idf,
        zone_name,
        proto,
        volume,
        infiltration_schedule,
        occ_schedule,
        light_schedule,
        equip_schedule,
        activity_schedule,
        clothing_schedule,
    )
    add_zone_hvac(idf, zone_name, heat_schedule, cool_schedule, control_schedule)
    add_output_requests(idf, zone_name)
    return idf.idfstr()
    


def run_energyplus(eplus_bin: Path, weather: Path, idf_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(eplus_bin),
        "-w", str(weather),
        "-d", str(output_dir),
        str(idf_path),
    ]
    subprocess.run(cmd, check=True)


def fetch_series(sqlite_path: Path, variable_name: str, key_value: str) -> pd.DataFrame:
    conn = sqlite3.connect(sqlite_path)

    def table_columns(table: str) -> set[str]:
        cur = conn.execute(f"PRAGMA table_info({table})")
        return {row[1] for row in cur.fetchall()}

    rdd_columns = table_columns("ReportDataDictionary")
    var_column = "VariableName" if "VariableName" in rdd_columns else "Name"

    query = f"""
        SELECT
            t.Month,
            rd.Value
        FROM ReportData rd
        JOIN ReportDataDictionary rdd
            ON rd.ReportDataDictionaryIndex = rdd.ReportDataDictionaryIndex
        JOIN Time t
            ON rd.TimeIndex = t.TimeIndex
        WHERE rdd.{var_column} = ?
          AND rdd.KeyValue = ?
    """
    df = pd.read_sql_query(query, conn, params=(variable_name, key_value))
    conn.close()
    if df.empty:
        df = pd.DataFrame({"Month": [], "Value": []})
    return df


def aggregate_energy(sqlite_path: Path, zone_name: str, proto: Prototype) -> Dict[str, float]:
    results: Dict[str, float] = {}
    ideal_key = f"IDEAL_{zone_name}"

    def summarize(variable: str, key_value: str, convert_factor: float = JOULE_TO_KWH) -> Tuple[float, Dict[int, float]]:
        df = fetch_series(sqlite_path, variable, key_value)
        df["Value"] = df["Value"] * convert_factor
        monthly = df.groupby("Month")["Value"].sum().to_dict()
        annual = float(df["Value"].sum()) if not df.empty else 0.0
        return annual, {m: monthly.get(m, 0.0) for m in range(1, 13)}

    cool_annual, cool_month = summarize("Zone Ideal Loads Supply Air Total Cooling Energy", ideal_key)
    heat_annual, heat_month = summarize("Zone Ideal Loads Supply Air Total Heating Energy", ideal_key)
    lights_annual, lights_month = summarize("Zone Lights Electricity Energy", zone_name)
    equip_annual, equip_month = summarize("Zone Electric Equipment Electricity Energy", zone_name)

    cooling_final = {m: val / proto.cooling_cop for m, val in cool_month.items()}
    heating_final = {m: val / proto.heating_cop for m, val in heat_month.items()}
    results["cooling_kwh_annual"] = sum(cooling_final.values())
    results["heating_kwh_annual"] = sum(heating_final.values())
    results["other_electricity_kwh_annual"] = (lights_annual + equip_annual)

    for month in range(1, 13):
        results[f"cooling_kwh_m{month:02d}"] = cooling_final.get(month, 0.0)
        results[f"heating_kwh_m{month:02d}"] = heating_final.get(month, 0.0)
        other_val = lights_month.get(month, 0.0) + equip_month.get(month, 0.0)
        results[f"other_electricity_kwh_m{month:02d}"] = other_val
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="EnergyPlus batch simulator for Xinwu buildings")
    parser.add_argument("--building-path", type=Path, default=BUILDING_PATH)
    parser.add_argument("--land-use-path", type=Path, default=LAND_USE_PATH)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--energyplus-root", type=Path, default=ENERGYPLUS_ROOT)
    parser.add_argument("--weather-file", type=Path, default=DEFAULT_WEATHER)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="Directory to store per-building EnergyPlus outputs",
    )
    parser.add_argument("--limit", type=int, help="Limit number of buildings processed")
    parser.add_argument("--building-ids", nargs="*", help="Optional list of building IDs to simulate")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output CSV")

    args = parser.parse_args()
    energyplus_bin = args.energyplus_root / "energyplus"
    if not energyplus_bin.exists():
        sys.exit(f"EnergyPlus binary not found at {energyplus_bin}")
    if not args.weather_file.exists():
        sys.exit(f"Weather file not found: {args.weather_file}")

    prototypes, type_map, land_map, default_proto = load_prototype_config(args.config)
    ENERGY_DIR.mkdir(parents=True, exist_ok=True)
    args.runs_dir.mkdir(parents=True, exist_ok=True)

    buildings = gpd.read_file(args.building_path)
    buildings = assign_land_use(buildings, args.land_use_path)
    if args.building_ids:
        bid_set = {float(bid) for bid in args.building_ids}
        buildings = buildings[buildings["id"].isin(bid_set)]
    if args.limit:
        buildings = buildings.head(args.limit)
    if buildings.empty:
        sys.exit("No buildings selected for simulation")

    records = []
    for _, row in buildings.iterrows():
        proto_name = determine_prototype(row, type_map, land_map, default_proto)
        proto = prototypes[proto_name]
        b_id = row.get("id")
        if pd.isna(b_id):
            b_id = f"row_{int(row.name)}"
        else:
            b_id = str(int(b_id)) if float(b_id).is_integer() else str(b_id)
        area = row.get("footprint_area_m2", MIN_AREA_M2)
        height = row.get("height_m", MIN_HEIGHT_M)
        building_label, zone_name = make_labels(b_id)
        records.append((b_id, building_label, zone_name, area, height, proto))

    outputs = []
    for idx, (b_id, building_label, zone_name, area, height, proto) in enumerate(records, start=1):
        print(f"[{idx}/{len(records)}] Simulating building {b_id} with prototype {proto.name}...")
        idf_text = build_idf(b_id, building_label, zone_name, area, height, proto)
        run_dir = args.runs_dir / f"{b_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        idf_path = run_dir / "in.idf"
        with idf_path.open("w", encoding="utf-8") as f:
            f.write(idf_text)
        try:
            run_energyplus(energyplus_bin, args.weather_file, idf_path, run_dir)
            sqlite_path = run_dir / "eplusout.sql"
            if not sqlite_path.exists():
                raise FileNotFoundError("eplusout.sql missing after simulation")
            energy_stats = aggregate_energy(sqlite_path, zone_name, proto)
            record = {
                "building_id": b_id,
                "prototype": proto.name,
                "status": "ok",
                "source_weather": str(args.weather_file),
            }
            record.update(energy_stats)
            outputs.append(record)
        except Exception as exc:  # pragma: no cover - runtime safety
            print(f"Simulation failed for building {b_id}: {exc}", file=sys.stderr)
            outputs.append({
                "building_id": b_id,
                "prototype": proto.name,
                "status": f"error: {exc}",
            })

    df = pd.DataFrame(outputs)
    if args.output_csv.exists() and not args.overwrite:
        existing = pd.read_csv(args.output_csv)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset=["building_id"], keep="last")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(df)} records to {args.output_csv}")


if __name__ == "__main__":
    main()
