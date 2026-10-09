#!/usr/bin/env python3
"""gazebo_object_injector - place and randomize models in a Gazebo world file.

    python3 gazebo_object_injector.py                     # editor window
    python3 gazebo_object_injector.py randomize track.world --seed 7
    python3 gazebo_object_injector.py --help              # all commands

Needs Python 3.8+ and PyYAML (for rules files). The editor needs tkinter,
the preview command needs matplotlib. See README.md.
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import math
import os
import random
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # headless install: everything except the editor works
    tk = None

__version__ = "0.1.0"


# ======================================================================
# World file: marker blocks, spawn points, objects
# ======================================================================


NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MODEL_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]*$")

# Pose written by older versions of this tool for objects that found no spot.
LEGACY_HIDDEN_Z = -500.0


class WorldError(ValueError):
    """Raised for invalid input or a world file this module cannot edit."""


def validate_name(name: str, what: str = "name") -> str:
    if not NAME_RE.match(name or ""):
        raise WorldError(
            f"invalid {what} {name!r}: use letters, digits and '_' "
            "and do not start with a digit"
        )
    return name


def validate_model(model: str) -> str:
    if not MODEL_RE.match(model or "") or "--" in model:
        raise WorldError(
            f"invalid model {model!r}: use letters, digits, '_', '-', '.', '/' "
            "and no '--' (it is not allowed inside XML comments)"
        )
    return model


def _fmt(v: float) -> str:
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


@dataclass(frozen=True)
class Pose:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    @classmethod
    def parse(cls, text: str) -> "Pose":
        parts = text.split()
        if not 3 <= len(parts) <= 6:
            raise WorldError(f"pose needs 3 or 6 numbers, got {text!r}")
        try:
            vals = [float(p) for p in parts]
        except ValueError as exc:
            raise WorldError(f"pose is not numeric: {text!r}") from exc
        vals += [0.0] * (6 - len(vals))
        return cls(*vals)

    def __str__(self) -> str:
        return " ".join(_fmt(v) for v in (self.x, self.y, self.z, self.roll, self.pitch, self.yaw))

    def distance_xy(self, other: "Pose") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def is_close(self, other: "Pose", tol: float = 1e-3) -> bool:
        a = (self.x, self.y, self.z, self.roll, self.pitch, self.yaw)
        b = (other.x, other.y, other.z, other.roll, other.pitch, other.yaw)
        return all(abs(p - q) < tol for p, q in zip(a, b))


@dataclass
class WorldObject:
    """One injected <include>.

    random=False: the object stays at ``pose``.
    random=True:  ``pool`` lists the spawn points the randomizer may use.
    ``at`` is the spawn point the object currently occupies ("" if none),
    ``placed`` is False when the object is currently left out of the world.
    """

    name: str
    model: str
    pose: Pose = field(default_factory=Pose)
    random: bool = False
    pool: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    at: str = ""
    placed: bool = True

    def validate(self) -> None:
        validate_name(self.name, "object name")
        validate_model(self.model)
        for t in self.tags:
            validate_name(t, "tag")
        for s in self.pool:
            validate_name(s, "spawn point name")
        if self.random and not self.pool:
            raise WorldError(f"random object {self.name!r} needs at least one spawn point")


_BLOCK_TAG = "object-injector"
_SPAWN_RE = re.compile(r"(?P<indent>[ \t]*)<!-- \[SPAWN-POINTS\](?P<body>.*?)\[/SPAWN-POINTS\] -->", re.DOTALL)
_START_RE = re.compile(r"<!-- \[" + _BLOCK_TAG + r":([^\]]+)\] -->")
_META_RE = re.compile(r"<!-- META: (.*?) -->")
_POSE_RE = re.compile(r"<pose>([^<]*)</pose>")


def _block_re(name: str) -> "re.Pattern[str]":
    n = re.escape(name)
    return re.compile(
        r"(?P<indent>[ \t]*)<!-- \[" + _BLOCK_TAG + ":" + n + r"\] -->.*?"
        r"<!-- \[/" + _BLOCK_TAG + ":" + n + r"\] -->",
        re.DOTALL,
    )


def _render_block(obj: WorldObject, indent: str) -> str:
    meta = [
        f"uri={obj.model}",
        f"random={obj.random}",
        f"spawns={','.join(obj.pool)}",
        f"tags={','.join(obj.tags)}",
        f"at={obj.at}",
    ]
    if not obj.placed:
        meta.append(f"pose={obj.pose}")
    lines = [
        f"<!-- [{_BLOCK_TAG}:{obj.name}] -->",
        f"<!-- META: {' | '.join(meta)} -->",
    ]
    if obj.placed:
        lines += [
            "<include>",
            f"  <name>{obj.name}</name>",
            f"  <uri>model://{obj.model}</uri>",
            f"  <pose>{obj.pose}</pose>",
            "</include>",
        ]
    lines.append(f"<!-- [/{_BLOCK_TAG}:{obj.name}] -->")
    return "\n".join(indent + ln for ln in lines)


def _parse_block(name: str, block: str) -> WorldObject:
    meta: Dict[str, str] = {}
    m = _META_RE.search(block)
    if m:
        for part in m.group(1).split(" | "):
            if "=" in part:
                k, v = part.split("=", 1)
                meta[k.strip()] = v.strip()
    has_include = "<include>" in block
    pm = _POSE_RE.search(block)
    if pm:
        pose = Pose.parse(pm.group(1))
    elif "pose" in meta:
        pose = Pose.parse(meta["pose"])
    else:
        pose = Pose()
    placed = has_include and pose.z > LEGACY_HIDDEN_Z + 1.0
    model = meta.get("uri", "")
    if not model:
        um = re.search(r"<uri>model://([^<]+)</uri>", block)
        model = um.group(1) if um else ""

    def _list(key: str) -> List[str]:
        return [s.strip() for s in meta.get(key, "").split(",") if s.strip()]

    return WorldObject(
        name=name,
        model=model,
        pose=pose,
        random=meta.get("random", "False") == "True",
        pool=_list("spawns"),
        tags=_list("tags"),
        at=meta.get("at", "") if placed else "",
        placed=placed,
    )


class World:
    """In-memory world file. Edit it, then call :meth:`save`."""

    def __init__(self, text: str, path: Optional[str] = None):
        if "</world>" not in text:
            raise WorldError("no </world> tag found; is this an SDF world file?")
        self.text = text
        self.path = path

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str) -> "World":
        with open(path, "r", encoding="utf-8") as f:
            return cls(f.read(), path)

    def save(self, path: Optional[str] = None) -> str:
        """Write atomically. Refuses to write XML that does not parse."""
        path = path or self.path
        if not path:
            raise WorldError("no output path")
        try:
            ET.fromstring(self.text)
        except ET.ParseError as exc:
            raise WorldError(f"refusing to write invalid XML: {exc}") from exc
        directory = os.path.dirname(os.path.abspath(path))
        fd, tmp = tempfile.mkstemp(prefix=".goi-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(self.text)
            if os.path.exists(path):
                os.chmod(tmp, os.stat(path).st_mode & 0o7777)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return path

    # ------------------------------------------------------------ helpers
    def _insert_before_world_end(self, block: str) -> None:
        idx = self.text.rfind("</world>")
        line_start = self.text.rfind("\n", 0, idx) + 1
        if self.text[line_start:idx].strip():
            # </world> shares its line with other content: insert right before it.
            self.text = self.text[:idx] + "\n" + block + "\n" + self.text[idx:]
            return
        self.text = self.text[:line_start] + block + "\n\n" + self.text[line_start:]

    def _child_indent(self) -> str:
        m = re.search(r"\n([ \t]*)<world\b", self.text)
        base = m.group(1) if m else ""
        return base + "  "

    def entity_names(self) -> set:
        """Names of every model/actor/light/include in the file."""
        t = self.text
        names = set(re.findall(r"<(?:model|actor|light)\s+name=\"([^\"]+)\"", t))
        for inc in re.findall(r"<include>(.*?)</include>", t, re.DOTALL):
            m = re.search(r"<name>([^<]+)</name>", inc)
            if m:
                names.add(m.group(1).strip())
        return names

    # --------------------------------------------------------- spawn points
    def spawn_points(self) -> Dict[str, Pose]:
        m = _SPAWN_RE.search(self.text)
        if not m:
            return {}
        pts: Dict[str, Pose] = {}
        for line in m.group("body").splitlines():
            line = line.strip()
            if ":" not in line:
                continue
            name, pose = line.split(":", 1)
            pts[name.strip()] = Pose.parse(pose)
        return pts

    def _write_spawn_points(self, pts: Dict[str, Pose]) -> None:
        m = _SPAWN_RE.search(self.text)
        indent = m.group("indent") if m else self._child_indent()
        body = "".join(f"{indent}{n}:{p}\n" for n, p in pts.items())
        block = f"{indent}<!-- [SPAWN-POINTS]\n{body}{indent}[/SPAWN-POINTS] -->"
        if m:
            if pts:
                self.text = self.text[: m.start()] + block + self.text[m.end():]
            else:
                start = self.text.rfind("\n", 0, m.start())
                self.text = self.text[: max(start, 0)] + self.text[m.end():]
        elif pts:
            self._insert_before_world_end(block)

    def set_spawn_point(self, name: str, pose: Pose) -> None:
        validate_name(name, "spawn point name")
        pts = self.spawn_points()
        pts[name] = pose
        self._write_spawn_points(pts)

    def rename_spawn_point(self, old: str, new: str) -> List[str]:
        """Rename and update every object pool that referenced it."""
        validate_name(new, "spawn point name")
        pts = self.spawn_points()
        if old not in pts:
            raise WorldError(f"no spawn point {old!r}")
        if new != old and new in pts:
            raise WorldError(f"spawn point {new!r} already exists")
        pts = {(new if k == old else k): v for k, v in pts.items()}
        self._write_spawn_points(pts)
        changed = []
        for obj in self.objects().values():
            if old in obj.pool or obj.at == old:
                obj.pool = [new if s == old else s for s in obj.pool]
                if obj.at == old:
                    obj.at = new
                self._replace_block(obj.name, obj)
                changed.append(obj.name)
        return changed

    def remove_spawn_point(self, name: str) -> List[str]:
        """Remove and drop it from every pool. Returns the affected objects."""
        pts = self.spawn_points()
        if name not in pts:
            raise WorldError(f"no spawn point {name!r}")
        del pts[name]
        self._write_spawn_points(pts)
        changed = []
        for obj in self.objects().values():
            if name in obj.pool or obj.at == name:
                obj.pool = [s for s in obj.pool if s != name]
                if obj.at == name:
                    obj.at = ""
                self._replace_block(obj.name, obj)
                changed.append(obj.name)
        return changed

    # -------------------------------------------------------------- objects
    def object_names(self) -> List[str]:
        seen: List[str] = []
        for n in _START_RE.findall(self.text):
            if n not in seen:
                seen.append(n)
        return seen

    def objects(self) -> Dict[str, WorldObject]:
        out: Dict[str, WorldObject] = {}
        for name in self.object_names():
            m = _block_re(name).search(self.text)
            if m:
                out[name] = _parse_block(name, m.group(0))
        return out

    def get(self, name: str) -> WorldObject:
        m = _block_re(name).search(self.text)
        if not m:
            raise WorldError(f"no object {name!r}")
        return _parse_block(name, m.group(0))

    def _replace_block(self, name: str, obj: WorldObject) -> None:
        m = _block_re(name).search(self.text)
        if not m:
            raise WorldError(f"no object {name!r}")
        self.text = self.text[: m.start()] + _render_block(obj, m.group("indent")) + self.text[m.end():]

    def _check_name_free(self, name: str) -> None:
        if name in self.object_names() or name in self.entity_names():
            raise WorldError(f"an entity named {name!r} already exists in the world")

    def _check_pool(self, obj: WorldObject) -> None:
        pts = self.spawn_points()
        missing = [s for s in obj.pool if s not in pts]
        if missing:
            raise WorldError(f"unknown spawn point(s): {', '.join(missing)}")

    def add_object(self, obj: WorldObject) -> None:
        obj.validate()
        self._check_name_free(obj.name)
        self._check_pool(obj)
        if obj.random and not obj.at:
            # Until the first randomization: first pool spawn point nobody uses yet.
            taken = {o.at for o in self.objects().values() if o.placed}
            obj.at = next((s for s in obj.pool if s not in taken), obj.pool[0])
            obj.pose = self.spawn_points()[obj.at]
        self._insert_before_world_end(_render_block(obj, self._child_indent()))

    def update_object(self, old_name: str, obj: WorldObject) -> None:
        """Replace an object in place (same position in the file)."""
        obj.validate()
        self.get(old_name)
        if obj.name != old_name:
            self._check_name_free(obj.name)
        self._check_pool(obj)
        if obj.random:
            pts = self.spawn_points()
            if obj.at not in obj.pool:
                obj.at = obj.pool[0]
            obj.pose = pts[obj.at]
            obj.placed = True
        else:
            obj.at = ""
            obj.placed = True
        self._replace_block(old_name, obj)

    def remove_object(self, name: str) -> None:
        m = _block_re(name).search(self.text)
        if not m:
            raise WorldError(f"no object {name!r}")
        start = self.text.rfind("\n", 0, m.start())
        end = m.end()
        if self.text[end:end + 2] == "\n\n":
            end += 1
        self.text = self.text[: max(start, 0)] + self.text[end:]

    def place_object(self, name: str, pose: Pose, at: str = "") -> None:
        obj = self.get(name)
        self._replace_block(name, replace(obj, pose=pose, at=at, placed=True))

    def unplace_object(self, name: str) -> None:
        obj = self.get(name)
        self._replace_block(name, replace(obj, at="", placed=False))

    # ------------------------------------------------------------ checking
    def problems(self) -> List[str]:
        """Human-readable list of inconsistencies (empty if none)."""
        out: List[str] = []
        pts = self.spawn_points()
        names = self.object_names()
        if len(names) != len(set(names)):
            out.append("duplicate object blocks in the file")
        others = self.entity_names() - set(names)
        for obj in self.objects().values():
            if obj.name in others:
                out.append(f"{obj.name}: name collides with another entity in the world")
            if not obj.model:
                out.append(f"{obj.name}: no model uri")
            for s in obj.pool:
                if s not in pts:
                    out.append(f"{obj.name}: pool references missing spawn point {s!r}")
            if obj.random and not obj.pool:
                out.append(f"{obj.name}: random object with an empty pool")
        return out


def iter_models(directories: Iterable[str]) -> List[str]:
    """Model directory names (folders with a model.config) found in ``directories``."""
    found = set()
    for d in directories:
        if not d or not os.path.isdir(d):
            continue
        for entry in os.listdir(d):
            if os.path.isfile(os.path.join(d, entry, "model.config")):
                found.add(entry)
    return sorted(found)


def model_search_path() -> List[str]:
    """Directories from GAZEBO_MODEL_PATH, GZ_SIM_RESOURCE_PATH and ~/.gazebo/models."""
    dirs: List[str] = []
    for var in ("GAZEBO_MODEL_PATH", "GZ_SIM_RESOURCE_PATH", "IGN_GAZEBO_RESOURCE_PATH"):
        dirs += [p for p in os.environ.get(var, "").split(os.pathsep) if p]
    dirs.append(os.path.expanduser("~/.gazebo/models"))
    out: List[str] = []
    for d in dirs:
        if d not in out:
            out.append(d)
    return out


# ======================================================================
# Randomizer: pools, seed, near / keep_clear rules
# ======================================================================




class RuleError(ValueError):
    """Invalid rules file or a rule marked ``required`` that could not be met."""


@dataclass(frozen=True)
class Selector:
    """Matches objects by tag, name pattern and/or model pattern (all must match)."""

    tag: Optional[str] = None
    name: Optional[str] = None
    model: Optional[str] = None

    def matches(self, obj: WorldObject) -> bool:
        if self.tag is not None and self.tag not in obj.tags:
            return False
        if self.name is not None and not fnmatch.fnmatchcase(obj.name, self.name):
            return False
        if self.model is not None and not fnmatch.fnmatchcase(obj.model, self.model):
            return False
        return True

    def __str__(self) -> str:
        parts = [f"{k}={v}" for k, v in (("tag", self.tag), ("name", self.name), ("model", self.model)) if v]
        return "{" + ", ".join(parts) + "}"


@dataclass
class Rule:
    kind: str  # "near" | "keep_clear"
    id: str
    targets: Selector
    objects: Optional[Selector] = None
    count: int = 1
    max_distance: float = math.inf
    radius: float = 0.0
    except_rules: List[str] = field(default_factory=list)
    required: bool = False


@dataclass
class RuleSet:
    rules: List[Rule] = field(default_factory=list)
    seed: Optional[int] = None


@dataclass
class RandomizeResult:
    seed: int
    placements: Dict[str, Optional[str]] = field(default_factory=dict)  # object -> spawn or None
    reserved: Dict[str, str] = field(default_factory=dict)  # spawn -> rule id
    links: List[Tuple[str, str, str]] = field(default_factory=list)  # (object, target, rule id)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------- parsing

_SELECTOR_KEYS = {"tag", "name", "model"}


def _selector(raw: Any, where: str) -> Selector:
    if not isinstance(raw, dict) or not raw:
        raise RuleError(f"{where}: expected a mapping like {{tag: obstacle}}")
    unknown = set(raw) - _SELECTOR_KEYS
    if unknown:
        raise RuleError(f"{where}: unknown key(s) {sorted(unknown)}; use tag, name, model")
    return Selector(**{k: str(v) for k, v in raw.items()})


def _number(raw: Dict[str, Any], key: str, where: str, default: Any = None, minimum: float = 0.0) -> Any:
    if key not in raw:
        if default is None:
            raise RuleError(f"{where}: '{key}' is required")
        return default
    try:
        v = float(raw[key])
    except (TypeError, ValueError) as exc:
        raise RuleError(f"{where}: '{key}' must be a number") from exc
    if v < minimum:
        raise RuleError(f"{where}: '{key}' must be >= {minimum}")
    return v


def parse_rules(data: Any) -> RuleSet:
    """Build a :class:`RuleSet` from already-loaded YAML/JSON data."""
    if data is None:
        return RuleSet()
    if not isinstance(data, dict):
        raise RuleError("rules file must be a mapping with a 'rules' list")
    unknown = set(data) - {"rules", "seed"}
    if unknown:
        raise RuleError(f"unknown top-level key(s) {sorted(unknown)}")
    seed = data.get("seed")
    if seed is not None and not isinstance(seed, int):
        raise RuleError("'seed' must be an integer")
    raw_rules = data.get("rules") or []
    if not isinstance(raw_rules, list):
        raise RuleError("'rules' must be a list")

    rules: List[Rule] = []
    ids: Set[str] = set()
    for i, raw in enumerate(raw_rules, 1):
        where = f"rule {i}"
        if not isinstance(raw, dict):
            raise RuleError(f"{where}: expected a mapping")
        kinds = [k for k in ("near", "keep_clear") if k in raw]
        extra = set(raw) - {"near", "keep_clear", "id", "required"}
        if len(kinds) != 1 or extra:
            raise RuleError(f"{where}: needs exactly one of 'near' / 'keep_clear' (plus optional id, required)")
        kind = kinds[0]
        rid = str(raw.get("id", f"{kind}_{i}"))
        if rid in ids:
            raise RuleError(f"{where}: duplicate id {rid!r}")
        ids.add(rid)
        body = raw[kind]
        if not isinstance(body, dict):
            raise RuleError(f"{where}: '{kind}' must be a mapping")
        where = f"rule {i} ({rid})"
        if kind == "near":
            allowed = {"objects", "targets", "count", "max_distance"}
            if set(body) - allowed:
                raise RuleError(f"{where}: unknown key(s) {sorted(set(body) - allowed)}")
            count = _number(body, "count", where, default=1, minimum=1)
            if count != int(count):
                raise RuleError(f"{where}: 'count' must be an integer")
            rules.append(Rule(
                kind=kind, id=rid,
                objects=_selector(body.get("objects"), f"{where}.objects"),
                targets=_selector(body.get("targets"), f"{where}.targets"),
                count=int(count),
                max_distance=_number(body, "max_distance", where, default=math.inf),
                required=bool(raw.get("required", False)),
            ))
        else:
            allowed = {"targets", "radius", "except"}
            if set(body) - allowed:
                raise RuleError(f"{where}: unknown key(s) {sorted(set(body) - allowed)}")
            exc = body.get("except", [])
            if isinstance(exc, str):
                exc = [exc]
            missing = [e for e in exc if e not in ids]
            if missing:
                raise RuleError(f"{where}: 'except' refers to unknown or later rule(s) {missing}")
            rules.append(Rule(
                kind=kind, id=rid,
                targets=_selector(body.get("targets"), f"{where}.targets"),
                radius=_number(body, "radius", where),
                except_rules=[str(e) for e in exc],
            ))
    return RuleSet(rules=rules, seed=seed)


def load_rules(path: str) -> RuleSet:
    import yaml  # imported lazily so the world editing API works without PyYAML

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise RuleError(f"{path}: {exc}") from exc
    try:
        return parse_rules(data)
    except RuleError as exc:
        raise RuleError(f"{path}: {exc}") from exc


def default_rules_path(world_path: str) -> Optional[str]:
    """``foo.world`` -> ``foo.rules.yaml`` next to it, if that file exists."""
    stem, _ = os.path.splitext(world_path)
    for cand in (stem + ".rules.yaml", stem + ".rules.yml"):
        if os.path.isfile(cand):
            return cand
    return None


# ----------------------------------------------------------- randomizing

def randomize(world: World, rules: Optional[RuleSet] = None, seed: Optional[int] = None) -> RandomizeResult:
    """Randomize ``world`` in memory. Fixed objects are never moved."""
    rules = rules or RuleSet()
    if seed is None:
        seed = rules.seed
    if seed is None:
        seed = random.SystemRandom().randrange(2**31)
    rng = random.Random(seed)
    res = RandomizeResult(seed=seed)

    spawns = world.spawn_points()
    objects = world.objects()
    movable = sorted(n for n, o in objects.items() if o.random)

    used: Set[str] = set()
    pose_of: Dict[str, Pose] = {}  # every object that is in the world after this run
    for n, o in objects.items():
        if not o.random and o.placed:
            pose_of[n] = o.pose
            for s, p in spawns.items():
                if p.distance_xy(o.pose) < 0.05:
                    used.add(s)

    def free(spawn: str) -> bool:
        return spawn in spawns and spawn not in used and spawn not in res.reserved

    def place(name: str, spawn: Optional[str]) -> None:
        res.placements[name] = spawn
        if spawn is not None:
            used.add(spawn)
            pose_of[name] = spawns[spawn]

    def place_random(name: str) -> None:
        options = [s for s in objects[name].pool if free(s)]
        place(name, rng.choice(options) if options else None)

    # 1. targets first
    target_sels = [r.targets for r in rules.rules]
    first = [n for n in movable if any(s.matches(objects[n]) for s in target_sels)]
    rng.shuffle(first)
    for n in first:
        place_random(n)

    # 2. rules
    rule_targets: Dict[str, Set[str]] = {}
    for rule in rules.rules:
        hits: Set[str] = set()
        rule_targets[rule.id] = hits
        targets = sorted(n for n in pose_of if rule.targets.matches(objects[n]))
        rng.shuffle(targets)

        if rule.kind == "near":
            assert rule.objects is not None
            subjects = sorted(n for n in movable if n not in res.placements and rule.objects.matches(objects[n]))
            rng.shuffle(subjects)
            done = 0
            for tgt in targets:
                if done == rule.count:
                    break
                tpose = pose_of[tgt]
                for subj in subjects:
                    if subj in res.placements:
                        continue
                    cands = [s for s in objects[subj].pool
                             if free(s) and spawns[s].distance_xy(tpose) <= rule.max_distance]
                    if not cands:
                        continue
                    best = min(cands, key=lambda s: (spawns[s].distance_xy(tpose), s))
                    place(subj, best)
                    res.links.append((subj, tgt, rule.id))
                    hits.add(tgt)
                    done += 1
                    break
            if done < rule.count:
                msg = (f"rule {rule.id}: placed {done} of {rule.count} {rule.objects} "
                       f"near {rule.targets}")
                if rule.required:
                    raise RuleError(msg)
                res.warnings.append(msg)

        else:  # keep_clear
            skip: Set[str] = set()
            for rid in rule.except_rules:
                skip |= rule_targets.get(rid, set())
            for tgt in targets:
                if tgt in skip:
                    continue
                hits.add(tgt)
                for s in sorted(spawns):
                    if s not in used and s not in res.reserved \
                            and spawns[s].distance_xy(pose_of[tgt]) <= rule.radius:
                        res.reserved[s] = rule.id

    # 3. everything else
    rest = [n for n in movable if n not in res.placements]
    rng.shuffle(rest)
    for n in rest:
        place_random(n)

    # write back
    for n in movable:
        spawn = res.placements.get(n)
        if spawn is None:
            world.unplace_object(n)
            if objects[n].pool:
                res.warnings.append(f"{n}: no free spawn point, left out of the world")
        else:
            world.place_object(n, spawns[spawn], at=spawn)
    return res


def randomize_file(
    world_path: str,
    output: Optional[str] = None,
    rules_path: Optional[str] = None,
    seed: Optional[int] = None,
    use_default_rules: bool = True,
) -> RandomizeResult:
    """Load, randomize and save. ``output`` defaults to overwriting ``world_path``."""
    world = World.load(world_path)
    if rules_path is None and use_default_rules:
        rules_path = default_rules_path(world_path)
    rules = load_rules(rules_path) if rules_path else RuleSet()
    result = randomize(world, rules, seed)
    world.save(output or world_path)
    return result


# ======================================================================
# Preview: top-down plot of a few seeds (matplotlib)
# ======================================================================


_THEMES = {
    "light": {
        "surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
        "spawn": "#9a9993", "reserved": "#c03a3a", "link": "#52514e",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    },
    "dark": {
        "surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#33332f",
        "spawn": "#7d7c76", "reserved": "#e66767", "link": "#c3c2b7",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
    },
}
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]


def render_preview(
    world_path: str,
    rules: Optional[RuleSet],
    seeds: Sequence[Optional[int]],
    output: str,
    title: Optional[str] = None,
    theme: str = "light",
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    th = _THEMES[theme]
    base = World.load(world_path)
    spawns = base.spawn_points()

    # Color follows the tag, in order of first appearance in the file.
    tag_order: List[str] = []
    for o in base.objects().values():
        key = o.tags[0] if o.tags else "untagged"
        if key not in tag_order:
            tag_order.append(key)

    def style(tag: str):
        i = tag_order.index(tag)
        color = th["series"][i] if i < len(th["series"]) else th["muted"]
        return color, _MARKERS[i % len(_MARKERS)]

    xs = [p.x for p in spawns.values()] + [o.pose.x for o in base.objects().values() if not o.random]
    ys = [p.y for p in spawns.values()] + [o.pose.y for o in base.objects().values() if not o.random]
    if not xs:
        xs, ys = [0.0], [0.0]
    pad = 1.5
    xlim = (min(xs) - pad, max(xs) + pad)
    ylim = (min(ys) - pad, max(ys) + pad)
    aspect = (ylim[1] - ylim[0]) / max(xlim[1] - xlim[0], 1e-6)

    n = len(seeds)
    cols = min(n, 3)
    rows = math.ceil(n / cols)
    panel_w = 5.2
    fig, axes = plt.subplots(rows, cols, figsize=(panel_w * cols, panel_w * aspect * rows + 1.1),
                             squeeze=False, facecolor=th["surface"])

    for idx, seed in enumerate(seeds):
        ax = axes[idx // cols][idx % cols]
        w = World(base.text, base.path)
        res = randomize(w, rules, seed)
        objs = w.objects()

        ax.set_facecolor(th["surface"])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.grid(True, color=th["grid"], linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=th["muted"], labelsize=7, length=0)
        ax.set_title(f"seed {res.seed}", color=th["text"], fontsize=10, loc="left")

        for name, p in spawns.items():
            reserved = name in res.reserved
            ax.scatter(p.x, p.y, s=70, facecolors="none", linewidths=1.0,
                       edgecolors=th["reserved"] if reserved else th["spawn"],
                       linestyle="--" if reserved else "-", zorder=2)
            ax.annotate(name, (p.x, p.y), xytext=(0, -11), textcoords="offset points",
                        ha="center", va="top", fontsize=6, color=th["muted"])
            if reserved:
                ax.plot([p.x - 0.18, p.x + 0.18], [p.y - 0.18, p.y + 0.18], color=th["reserved"], lw=1)
                ax.plot([p.x - 0.18, p.x + 0.18], [p.y + 0.18, p.y - 0.18], color=th["reserved"], lw=1)

        for subj, tgt, _ in res.links:
            a, b = objs[tgt].pose, objs[subj].pose
            ax.annotate("", xy=(b.x, b.y), xytext=(a.x, a.y),
                        arrowprops=dict(arrowstyle="->", color=th["link"], lw=1, linestyle=(0, (3, 2)),
                                        shrinkA=7, shrinkB=7), zorder=3)

        for o in objs.values():
            if not o.placed:
                continue
            color, marker = style(o.tags[0] if o.tags else "untagged")
            ax.scatter(o.pose.x, o.pose.y, s=64, c=color, marker=marker,
                       edgecolors=th["surface"], linewidths=1.5, zorder=4)
            ax.annotate(o.name, (o.pose.x, o.pose.y), xytext=(7, 5), textcoords="offset points",
                        ha="left", fontsize=7, color=th["text"], zorder=5)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    handles = []
    for tag in tag_order:
        color, marker = style(tag)
        handles.append(Line2D([], [], marker=marker, linestyle="", color=color,
                              markeredgecolor=th["surface"], markersize=8, label=tag))
    handles.append(Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                          markeredgecolor=th["spawn"], markersize=8, label="spawn point"))
    handles.append(Line2D([], [], marker="x", linestyle="", color=th["reserved"],
                          markersize=7, label="kept clear"))
    if rules and any(r.kind == "near" for r in rules.rules):
        handles.append(Line2D([], [], linestyle=(0, (3, 2)), color=th["link"], label="placed near"))
    leg = fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
                     fontsize=8, labelcolor=th["text"])
    leg.set_in_layout(True)
    if title:
        fig.suptitle(title, color=th["text"], fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output, dpi=160, facecolor=th["surface"])
    plt.close(fig)
    return output


# ======================================================================
# Editor window (tkinter)
# ======================================================================


_CACHE = os.path.expanduser("~/.cache/gazebo_object_injector/last_world")
_POSE_LABELS = ("x", "y", "z", "roll", "pitch", "yaw")


def _remember(path: str) -> None:
    try:
        os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
        with open(_CACHE, "w", encoding="utf-8") as f:
            f.write(path)
    except OSError:
        pass


def _last_world() -> str:
    try:
        with open(_CACHE, encoding="utf-8") as f:
            p = f.read().strip()
        return p if os.path.isfile(p) else ""
    except OSError:
        return ""


class PoseEntry(ttk.Frame if tk else object):
    def __init__(self, master: tk.Misc):
        super().__init__(master)
        self.vars = [tk.StringVar(value="0") for _ in _POSE_LABELS]
        for i, (lbl, var) in enumerate(zip(_POSE_LABELS, self.vars)):
            ttk.Label(self, text=lbl).grid(row=i // 3 * 2, column=i % 3, sticky="w", padx=(0, 6))
            ttk.Entry(self, textvariable=var, width=9).grid(row=i // 3 * 2 + 1, column=i % 3, padx=(0, 6), pady=(0, 4))

    def get(self) -> Pose:
        try:
            return Pose(*[float(v.get()) for v in self.vars])
        except ValueError as exc:
            raise WorldError("pose fields must be numbers") from exc

    def set(self, pose: Pose) -> None:
        for v, val in zip(self.vars, (pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw)):
            v.set(f"{val:g}")


class App:
    def __init__(self, root: tk.Tk, world: Optional[str], model_dirs: List[str]):
        self.root = root
        root.title("Gazebo Object Injector")
        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        self.world_var = tk.StringVar(value=world or _last_world())
        self.model_dirs = list(model_dirs) + model_search_path()
        self.status = tk.StringVar()

        top = ttk.Frame(root, padding=(12, 12, 12, 0))
        top.pack(fill="x")
        ttk.Label(top, text="World").pack(side="left")
        ttk.Entry(top, textvariable=self.world_var, width=60).pack(side="left", padx=6, fill="x", expand=True)
        ttk.Button(top, text="Open…", command=self.open_world).pack(side="left")
        ttk.Button(top, text="Reload", command=self.refresh).pack(side="left", padx=(6, 0))
        ttk.Button(top, text="Model dir…", command=self.add_model_dir).pack(side="left", padx=(6, 0))

        nb = ttk.Notebook(root, padding=12)
        nb.pack(fill="both", expand=True)
        self._build_objects(ttk.Frame(nb, padding=8))
        nb.add(self.obj_tab, text="Objects")
        self._build_spawns(ttk.Frame(nb, padding=8))
        nb.add(self.sp_tab, text="Spawn points")
        self._build_randomize(ttk.Frame(nb, padding=8))
        nb.add(self.rnd_tab, text="Randomize")

        ttk.Label(root, textvariable=self.status, anchor="w", padding=(12, 0, 12, 8)).pack(fill="x")
        self.refresh()

    # ------------------------------------------------------------ plumbing
    def _world(self) -> World:
        path = self.world_var.get().strip()
        if not path or not os.path.isfile(path):
            raise WorldError("choose a world file first")
        return World.load(path)

    def _do(self, action: Callable[[World], Optional[str]]) -> None:
        """Load, apply ``action``, save, refresh. Errors leave the file untouched."""
        try:
            w = self._world()
            msg = action(w)
            w.save()
        except (WorldError, RuleError, OSError) as exc:
            messagebox.showerror("Error", str(exc), parent=self.root)
            return
        self.refresh()
        if msg:
            self.status.set(msg)

    def open_world(self) -> None:
        path = filedialog.askopenfilename(
            title="World file", filetypes=[("SDF world", "*.world *.sdf"), ("All files", "*")])
        if path:
            self.world_var.set(path)
            _remember(path)
            self.refresh()

    def add_model_dir(self) -> None:
        d = filedialog.askdirectory(title="Model directory")
        if d:
            self.model_dirs.insert(0, d)
            self._refresh_models()

    def _refresh_models(self) -> None:
        dirs = list(self.model_dirs)
        path = self.world_var.get().strip()
        if path:
            dirs.insert(0, os.path.join(os.path.dirname(os.path.abspath(path)), "models"))
        self.model_box["values"] = iter_models(dirs)

    def refresh(self) -> None:
        self._refresh_models()
        for tree in (self.obj_tree, self.sp_tree):
            tree.delete(*tree.get_children())
        self.pool_list.delete(0, "end")
        try:
            w = self._world()
        except (WorldError, OSError) as exc:
            self.status.set(str(exc))
            return
        if w.path:
            _remember(os.path.abspath(w.path))
        for name, p in w.spawn_points().items():
            self.sp_tree.insert("", "end", iid=name, values=(name, str(p)))
            self.pool_list.insert("end", name)
        for o in w.objects().values():
            mode = "random" if o.random else "fixed"
            at = (o.at if o.placed else "not placed") if o.random else str(o.pose)
            self.obj_tree.insert("", "end", iid=o.name, values=(o.name, o.model, ",".join(o.tags), mode, at))
        problems = w.problems()
        self.status.set(problems[0] if problems else f"{len(w.objects())} objects, "
                        f"{len(w.spawn_points())} spawn points")

    # ------------------------------------------------------------- objects
    def _build_objects(self, tab: ttk.Frame) -> None:
        self.obj_tab = tab
        cols = ("name", "model", "tags", "mode", "at")
        self.obj_tree = ttk.Treeview(tab, columns=cols, show="headings", height=14, selectmode="browse")
        for c, w in zip(cols, (120, 120, 110, 70, 150)):
            self.obj_tree.heading(c, text=c.capitalize())
            self.obj_tree.column(c, width=w, minwidth=50)
        self.obj_tree.grid(row=0, column=0, sticky="nsew")
        self.obj_tree.bind("<<TreeviewSelect>>", self._obj_selected)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        form = ttk.Frame(tab, padding=(12, 0, 0, 0))
        form.grid(row=0, column=1, sticky="n")
        self.o_name, self.o_model, self.o_tags = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.o_random = tk.BooleanVar(value=False)
        for r, (lbl, var) in enumerate((("Name", self.o_name), ("Tags", self.o_tags))):
            ttk.Label(form, text=lbl).grid(row=r * 2, column=0, sticky="w")
            ttk.Entry(form, textvariable=var, width=28).grid(row=r * 2 + 1, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(form, text="Model").grid(row=4, column=0, sticky="w")
        self.model_box = ttk.Combobox(form, textvariable=self.o_model, width=26)
        self.model_box.grid(row=5, column=0, sticky="ew", pady=(0, 6))

        modes = ttk.Frame(form)
        modes.grid(row=6, column=0, sticky="w", pady=(4, 4))
        ttk.Radiobutton(modes, text="Fixed pose", variable=self.o_random, value=False,
                        command=self._mode_changed).pack(side="left")
        ttk.Radiobutton(modes, text="Random from pool", variable=self.o_random, value=True,
                        command=self._mode_changed).pack(side="left", padx=(8, 0))

        self.pose_box = ttk.LabelFrame(form, text="Pose", padding=6)
        self.o_pose = PoseEntry(self.pose_box)
        self.o_pose.pack()
        self.pool_box = ttk.LabelFrame(form, text="Pool (select spawn points)", padding=6)
        self.pool_list = tk.Listbox(self.pool_box, selectmode="multiple", height=7, exportselection=False)
        self.pool_list.pack(fill="both")
        self.pose_box.grid(row=7, column=0, sticky="ew")

        btns = ttk.Frame(form)
        btns.grid(row=8, column=0, pady=(10, 0), sticky="ew")
        ttk.Button(btns, text="Add", command=self.obj_add).pack(side="left")
        ttk.Button(btns, text="Update", command=self.obj_update).pack(side="left", padx=6)
        ttk.Button(btns, text="Delete", command=self.obj_delete).pack(side="left")

    def _mode_changed(self) -> None:
        if self.o_random.get():
            self.pose_box.grid_forget()
            self.pool_box.grid(row=7, column=0, sticky="ew")
        else:
            self.pool_box.grid_forget()
            self.pose_box.grid(row=7, column=0, sticky="ew")

    def _obj_selected(self, _event: object = None) -> None:
        sel = self.obj_tree.selection()
        if not sel:
            return
        try:
            o = self._world().get(sel[0])
        except (WorldError, OSError):
            return
        self.o_name.set(o.name)
        self.o_model.set(o.model)
        self.o_tags.set(",".join(o.tags))
        self.o_random.set(o.random)
        self.o_pose.set(o.pose)
        self.pool_list.selection_clear(0, "end")
        for i in range(self.pool_list.size()):
            if self.pool_list.get(i) in o.pool:
                self.pool_list.selection_set(i)
        self._mode_changed()

    def _form_object(self) -> WorldObject:
        random_ = self.o_random.get()
        pool = [self.pool_list.get(i) for i in self.pool_list.curselection()] if random_ else []
        return WorldObject(
            name=self.o_name.get().strip(),
            model=self.o_model.get().strip(),
            pose=Pose() if random_ else self.o_pose.get(),
            random=random_,
            pool=pool,
            tags=[t.strip() for t in self.o_tags.get().split(",") if t.strip()],
        )

    def obj_add(self) -> None:
        def act(w: World) -> str:
            obj = self._form_object()
            w.add_object(obj)
            return f"added {obj.name}"
        self._do(act)

    def obj_update(self) -> None:
        sel = self.obj_tree.selection()
        if not sel:
            messagebox.showinfo("Update", "Select an object in the list first.", parent=self.root)
            return

        def act(w: World) -> str:
            obj = self._form_object()
            obj.at = w.get(sel[0]).at
            w.update_object(sel[0], obj)
            return f"updated {obj.name}"
        self._do(act)

    def obj_delete(self) -> None:
        sel = self.obj_tree.selection()
        if sel and messagebox.askyesno("Delete", f"Remove {sel[0]} from the world?", parent=self.root):
            self._do(lambda w: (w.remove_object(sel[0]), f"removed {sel[0]}")[1])

    # -------------------------------------------------------- spawn points
    def _build_spawns(self, tab: ttk.Frame) -> None:
        self.sp_tab = tab
        self.sp_tree = ttk.Treeview(tab, columns=("name", "pose"), show="headings", height=14, selectmode="browse")
        self.sp_tree.heading("name", text="Name")
        self.sp_tree.heading("pose", text="Pose (x y z roll pitch yaw)")
        self.sp_tree.column("name", width=130)
        self.sp_tree.column("pose", width=300)
        self.sp_tree.grid(row=0, column=0, sticky="nsew")
        self.sp_tree.bind("<<TreeviewSelect>>", self._sp_selected)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        form = ttk.Frame(tab, padding=(12, 0, 0, 0))
        form.grid(row=0, column=1, sticky="n")
        self.s_name = tk.StringVar()
        ttk.Label(form, text="Name").grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=self.s_name, width=28).grid(row=1, column=0, sticky="ew", pady=(0, 6))
        box = ttk.LabelFrame(form, text="Pose", padding=6)
        box.grid(row=2, column=0, sticky="ew")
        self.s_pose = PoseEntry(box)
        self.s_pose.pack()
        btns = ttk.Frame(form)
        btns.grid(row=3, column=0, pady=(10, 0), sticky="ew")
        ttk.Button(btns, text="Set", command=self.sp_set).pack(side="left")
        ttk.Button(btns, text="Rename", command=self.sp_rename).pack(side="left", padx=6)
        ttk.Button(btns, text="Delete", command=self.sp_delete).pack(side="left")
        ttk.Label(form, text="Set adds a new point or moves an existing one.\n"
                             "Rename and Delete also update object pools.",
                  foreground="#666", justify="left").grid(row=4, column=0, sticky="w", pady=(10, 0))

    def _sp_selected(self, _event: object = None) -> None:
        sel = self.sp_tree.selection()
        if not sel:
            return
        try:
            pose = self._world().spawn_points()[sel[0]]
        except (WorldError, OSError, KeyError):
            return
        self.s_name.set(sel[0])
        self.s_pose.set(pose)

    def sp_set(self) -> None:
        def act(w: World) -> str:
            name = self.s_name.get().strip()
            w.set_spawn_point(name, self.s_pose.get())
            return f"set {name}"
        self._do(act)

    def sp_rename(self) -> None:
        sel = self.sp_tree.selection()
        if not sel:
            return

        def act(w: World) -> str:
            new = self.s_name.get().strip()
            changed = w.rename_spawn_point(sel[0], new)
            return f"renamed {sel[0]} to {new}" + (f", updated {len(changed)} objects" if changed else "")
        self._do(act)

    def sp_delete(self) -> None:
        sel = self.sp_tree.selection()
        if not sel or not messagebox.askyesno("Delete", f"Delete spawn point {sel[0]}?", parent=self.root):
            return

        def act(w: World) -> str:
            changed = w.remove_spawn_point(sel[0])
            return f"deleted {sel[0]}" + (f", removed from {len(changed)} pools" if changed else "")
        self._do(act)

    # ----------------------------------------------------------- randomize
    def _build_randomize(self, tab: ttk.Frame) -> None:
        self.rnd_tab = tab
        self.r_rules, self.r_seed, self.r_out = tk.StringVar(), tk.StringVar(), tk.StringVar()
        rows = (("Rules file", self.r_rules, "empty: <world>.rules.yaml if it exists"),
                ("Seed", self.r_seed, "empty: new random seed"),
                ("Output", self.r_out, "empty: overwrite the world file"))
        for r, (lbl, var, hint) in enumerate(rows):
            ttk.Label(tab, text=lbl).grid(row=r, column=0, sticky="w", pady=3)
            ttk.Entry(tab, textvariable=var, width=50).grid(row=r, column=1, sticky="ew", padx=6)
            ttk.Label(tab, text=hint, foreground="#666").grid(row=r, column=2, sticky="w")
        tab.columnconfigure(1, weight=1)
        btns = ttk.Frame(tab)
        btns.grid(row=3, column=1, sticky="w", pady=8, padx=6)
        ttk.Button(btns, text="Randomize", command=self.run_randomize).pack(side="left")
        ttk.Button(btns, text="Check", command=self.run_check).pack(side="left", padx=6)
        ttk.Button(btns, text="Preview 3 seeds", command=self.run_preview).pack(side="left")
        self.log = tk.Text(tab, height=14, width=90, font="TkFixedFont", relief="flat")
        self.log.grid(row=4, column=0, columnspan=3, sticky="nsew")
        tab.rowconfigure(4, weight=1)

    def _cli(self, *argv: str) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            main(list(argv))
        self.log.delete("1.0", "end")
        self.log.insert("1.0", buf.getvalue())
        self.refresh()

    def _rule_args(self) -> List[str]:
        r = self.r_rules.get().strip()
        return ["--rules", r] if r else []

    def run_randomize(self) -> None:
        args = ["randomize", self.world_var.get().strip(), *self._rule_args()]
        if self.r_seed.get().strip():
            args += ["--seed", self.r_seed.get().strip()]
        if self.r_out.get().strip():
            args += ["-o", self.r_out.get().strip()]
        self._cli(*args)

    def run_check(self) -> None:
        self._cli("check", self.world_var.get().strip(), *self._rule_args())

    def run_preview(self) -> None:
        out = os.path.join(tempfile.gettempdir(), "object_injector_preview.png")
        self._cli("preview", self.world_var.get().strip(), *self._rule_args(), "-o", out)
        if not os.path.isfile(out):
            return
        win = tk.Toplevel(self.root)
        win.title("Preview")
        img = tk.PhotoImage(file=out)
        scale = max(1, -(-img.width() // 1400))
        if scale > 1:
            img = img.subsample(scale)
        lbl = ttk.Label(win, image=img)
        lbl.image = img  # keep a reference
        lbl.pack()


def run_gui(world: Optional[str] = None, model_dirs: Optional[List[str]] = None) -> int:
    if tk is None:
        raise WorldError("tkinter is not installed (on Ubuntu: sudo apt install python3-tk)")
    root = tk.Tk()
    App(root, world, model_dirs or [])
    root.mainloop()
    return 0


# ======================================================================
# Command line
# ======================================================================



def _csv(text: Optional[str]) -> List[str]:
    return [s.strip() for s in (text or "").split(",") if s.strip()]


def _pose(values: Optional[Sequence[str]]) -> Optional[Pose]:
    if values is None:
        return None
    if len(values) not in (3, 6):
        raise WorldError("--pose takes X Y Z or X Y Z ROLL PITCH YAW")
    return Pose.parse(" ".join(values))


def _table(rows: List[List[str]], header: List[str]) -> str:
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    line = lambda r: "  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip()  # noqa: E731
    return "\n".join([line(header), line(["-" * w for w in widths])] + [line(r) for r in rows])


def _rules_for(args: argparse.Namespace) -> RuleSet:
    if getattr(args, "no_rules", False):
        return RuleSet()
    path = args.rules or default_rules_path(args.world)
    if path:
        if not args.rules:
            print(f"using rules from {path}", file=sys.stderr)
        return load_rules(path)
    return RuleSet()


# -------------------------------------------------------------- commands

def cmd_list(args: argparse.Namespace) -> int:
    w = World.load(args.world)
    objs = w.objects()
    pts = w.spawn_points()
    if objs:
        rows = []
        for o in objs.values():
            if o.random:
                where = o.at if o.placed else "(not placed)"
                kind = f"random [{','.join(o.pool)}]"
            else:
                where, kind = str(o.pose), "fixed"
            rows.append([o.name, o.model, ",".join(o.tags) or "-", kind, where])
        print(_table(rows, ["OBJECT", "MODEL", "TAGS", "MODE", "AT"]))
    else:
        print("no objects")
    print()
    if pts:
        print(_table([[n, str(p)] for n, p in pts.items()], ["SPAWN POINT", "POSE"]))
    else:
        print("no spawn points")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    w = World.load(args.world)
    pool = _csv(args.pool)
    pose = _pose(args.pose)
    if bool(pool) == bool(pose):
        raise WorldError("give either --pose (fixed object) or --pool (random object)")
    obj = WorldObject(name=args.name, model=args.model, pose=pose or Pose(),
                      random=bool(pool), pool=pool, tags=_csv(args.tags))
    w.add_object(obj)
    w.save()
    print(f"added {args.name}")
    return 0


def cmd_edit(args: argparse.Namespace) -> int:
    w = World.load(args.world)
    obj = w.get(args.name)
    if args.new_name:
        obj.name = args.new_name
    if args.model:
        obj.model = args.model
    if args.tags is not None:
        obj.tags = _csv(args.tags)
    pose = _pose(args.pose)
    if pose is not None:
        obj.random, obj.pool, obj.pose = False, [], pose
    if args.pool is not None:
        obj.random, obj.pool = True, _csv(args.pool)
    w.update_object(args.name, obj)
    w.save()
    print(f"updated {obj.name}")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    w = World.load(args.world)
    for n in args.names:
        w.remove_object(n)
    w.save()
    print(f"removed {', '.join(args.names)}")
    return 0


def cmd_spawn(args: argparse.Namespace) -> int:
    w = World.load(args.world)
    if args.spawn_cmd == "set":
        w.set_spawn_point(args.name, _pose(args.pose))
        msg = f"set {args.name}"
    elif args.spawn_cmd == "rename":
        changed = w.rename_spawn_point(args.old, args.new)
        msg = f"renamed {args.old} -> {args.new}" + (f" (updated {', '.join(changed)})" if changed else "")
    else:
        changed = w.remove_spawn_point(args.name)
        msg = f"removed {args.name}" + (f" (dropped from {', '.join(changed)})" if changed else "")
    w.save()
    print(msg)
    return 0


def cmd_randomize(args: argparse.Namespace) -> int:
    w = World.load(args.world)
    rules = _rules_for(args)
    res = randomize(w, rules, args.seed)
    out = w.save(args.output or args.world)
    rows = [[n, s or "(not placed)"] for n, s in sorted(res.placements.items())]
    if rows:
        print(_table(rows, ["OBJECT", "SPAWN POINT"]))
    for subj, tgt, rid in res.links:
        print(f"  {subj} placed near {tgt} ({rid})")
    if res.reserved:
        print(f"  kept clear: {', '.join(sorted(res.reserved))}")
    for msg in res.warnings:
        print(f"warning: {msg}", file=sys.stderr)
    print(f"seed {res.seed} -> {out}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    w = World.load(args.world)
    problems = w.problems()
    try:
        rules = _rules_for(args)
    except RuleError as exc:
        problems.append(str(exc))
        rules = RuleSet()
    objs = list(w.objects().values())
    for r in rules.rules:
        sels = [("targets", r.targets)] + ([("objects", r.objects)] if r.objects else [])
        for label, sel in sels:
            if not any(sel.matches(o) for o in objs):
                problems.append(f"rule {r.id}: {label} {sel} matches no object")
    for p in problems:
        print(p)
    if not problems:
        print("ok")
    return 1 if problems else 0


def cmd_preview(args: argparse.Namespace) -> int:
    rules = _rules_for(args)
    seeds = args.seed or [None] * args.count
    out = render_preview(args.world, rules, seeds, args.output, title=args.title, theme=args.theme)
    print(f"wrote {out}")
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    return run_gui(args.world, args.models)


# ------------------------------------------------------------------ main

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gazebo_object_injector.py",
        description="Place and randomize models in a Gazebo world file.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", metavar="COMMAND")

    s = sub.add_parser("list", help="show objects and spawn points")
    s.add_argument("world")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("add", help="add an object (fixed with --pose, random with --pool)")
    s.add_argument("world")
    s.add_argument("name")
    s.add_argument("model", help="model name as used in model://NAME")
    s.add_argument("--pose", nargs="+", metavar="V", help="X Y Z [ROLL PITCH YAW]")
    s.add_argument("--pool", help="comma separated spawn points the object may use")
    s.add_argument("--tags", help="comma separated tags used by rules")
    s.set_defaults(func=cmd_add)

    s = sub.add_parser("edit", help="change an object in place")
    s.add_argument("world")
    s.add_argument("name")
    s.add_argument("--name", dest="new_name", help="rename")
    s.add_argument("--model")
    s.add_argument("--pose", nargs="+", metavar="V", help="make fixed at X Y Z [ROLL PITCH YAW]")
    s.add_argument("--pool", help="make random with this pool")
    s.add_argument("--tags", help="replace tags ('' to clear)")
    s.set_defaults(func=cmd_edit)

    s = sub.add_parser("remove", help="remove objects")
    s.add_argument("world")
    s.add_argument("names", nargs="+")
    s.set_defaults(func=cmd_remove)

    s = sub.add_parser("spawn", help="edit spawn points")
    s.add_argument("world")
    ss = s.add_subparsers(dest="spawn_cmd", metavar="ACTION")
    ss.required = True
    a = ss.add_parser("set", help="add or move a spawn point")
    a.add_argument("name")
    a.add_argument("pose", nargs="+", metavar="V", help="X Y Z [ROLL PITCH YAW]")
    a = ss.add_parser("rename", help="rename and update every pool that uses it")
    a.add_argument("old")
    a.add_argument("new")
    a = ss.add_parser("remove", help="remove and drop it from every pool")
    a.add_argument("name")
    s.set_defaults(func=cmd_spawn)

    def rules_args(s: argparse.ArgumentParser) -> None:
        g = s.add_mutually_exclusive_group()
        g.add_argument("--rules", help="rules YAML (default: WORLD_STEM.rules.yaml if present)")
        g.add_argument("--no-rules", action="store_true", help="ignore any rules file")

    s = sub.add_parser("randomize", help="move random objects to new spawn points")
    s.add_argument("world")
    rules_args(s)
    s.add_argument("--seed", type=int, help="same seed, same result")
    s.add_argument("-o", "--output", help="write here instead of overwriting WORLD")
    s.set_defaults(func=cmd_randomize)

    s = sub.add_parser("check", help="report broken references and rules that match nothing")
    s.add_argument("world")
    rules_args(s)
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("preview", help="plot top-down randomization results to an image (needs matplotlib)")
    s.add_argument("world")
    rules_args(s)
    s.add_argument("--seed", type=int, action="append", help="seed to plot (repeatable)")
    s.add_argument("--count", type=int, default=3, help="number of random seeds if --seed is not given")
    s.add_argument("--title", help="figure title")
    s.add_argument("--theme", choices=["light", "dark"], default="light")
    s.add_argument("-o", "--output", required=True, help="output image (.png or .svg)")
    s.set_defaults(func=cmd_preview)

    s = sub.add_parser("gui", help="open the editor window (needs tkinter)")
    s.add_argument("world", nargs="?")
    s.add_argument("--models", action="append", default=[], metavar="DIR",
                   help="extra model directory (repeatable)")
    s.set_defaults(func=cmd_gui)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"list", "add", "edit", "remove", "spawn", "randomize", "check", "preview", "gui"}
    if not argv or (argv[0] not in commands and not argv[0].startswith("-")):
        argv = ["gui"] + argv  # no command: open the editor (optionally on the given world)
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (WorldError, RuleError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
