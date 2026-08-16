#!/usr/bin/env python3
"""Inspect and convert Condition Zero NEO rendering-map files.

This is a first-stage asset converter, not a complete BSP-to-VMF converter.
It exports the render geometry to Wavefront OBJ and extracts raw embedded
textures to TGA.  A Source/Hammer workflow can compile the OBJ through a
model toolchain and place the resulting MDL as a prop_static in a VMF.

The format information implemented here was recovered from engine_amd.so:
Neo_LoadNeoModel, loadFileStructure, and the individual lump loaders.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


HEADER_SIZE = 0x84
SUPPORTED_VERSIONS = range(1014, 1018)

LUMP_NAMES = (
    "vertices",
    "colors",
    "textures",
    "nodes",
    "texinfo",
    "meshes",
    "indices",
    "reserved_7",
    "draw_commands",
    "lights",
    "texcoords",
    "attributes",
    "fx_shaders",
    "fx_properties",
    "reserved_14",
    "models",
)

RECORD_SIZES = {
    0: 12,   # 3 little-endian floats
    1: 16,   # 4 little-endian floats
    3: 40,
    4: 12,
    5: 144,
    6: 4,    # uint32 index
    8: 12,   # uint32 GL mode, uint32 start, int32 count
    9: 84,
    10: 8,   # 2 little-endian floats
    11: 80,
    12: 132,
    15: 48,
}

# OpenGL primitive constants stored in the draw-command lump.
GL_POINTS = 0
GL_LINES = 1
GL_LINE_LOOP = 2
GL_LINE_STRIP = 3
GL_TRIANGLES = 4
GL_TRIANGLE_STRIP = 5
GL_TRIANGLE_FAN = 6
GL_QUADS = 7
GL_QUAD_STRIP = 8
GL_POLYGON = 9


class NeoFormatError(ValueError):
    pass


@dataclass(frozen=True)
class Lump:
    index: int
    name: str
    offset: int
    length: int
    record_size: int | None

    @property
    def count(self) -> int | None:
        if not self.record_size:
            return None
        return self.length // self.record_size


@dataclass(frozen=True)
class DrawCommand:
    mode: int
    start: int
    signed_count: int

    @property
    def indexed(self) -> bool:
        return self.signed_count < 0

    @property
    def count(self) -> int:
        return abs(self.signed_count)


@dataclass(frozen=True)
class TexInfo:
    uv_start: int
    texture_index: int
    flags: int


@dataclass(frozen=True)
class TextureInfo:
    index: int
    name: str
    width: int
    height: int
    pixel_size: int
    texture_format: int
    offset: int
    end: int
    external_reference: str | None = None


@dataclass(frozen=True)
class NeoLight:
    index: int
    position: tuple[float, float, float]
    intensity: float
    diffuse: tuple[float, float, float, float]
    secondary: tuple[float, float, float, float]
    ambient: tuple[float, float, float, float]
    reserved: tuple[float, float, float, float, float]


class NeoFile:
    def __init__(self, path: Path, data: bytes):
        self.path = path
        self.data = data
        if len(data) < HEADER_SIZE:
            raise NeoFormatError(f"{path}: file is smaller than the 132-byte header")

        self.version = struct.unpack_from("<I", data, 0)[0]
        self.lumps: list[Lump] = []
        for index, name in enumerate(LUMP_NAMES):
            offset, length = struct.unpack_from("<II", data, 4 + index * 8)
            lump = Lump(index, name, offset, length, RECORD_SIZES.get(index))
            self._validate_lump(lump)
            self.lumps.append(lump)

    @classmethod
    def read(cls, path: Path) -> "NeoFile":
        return cls(path, path.read_bytes())

    def _validate_lump(self, lump: Lump) -> None:
        if lump.length == 0:
            return
        if lump.offset < HEADER_SIZE:
            raise NeoFormatError(
                f"{self.path}: {lump.name} starts inside the header at 0x{lump.offset:x}"
            )
        if lump.offset + lump.length > len(self.data):
            raise NeoFormatError(
                f"{self.path}: {lump.name} ends beyond EOF "
                f"(0x{lump.offset + lump.length:x} > 0x{len(self.data):x})"
            )
        if lump.record_size and lump.length % lump.record_size:
            raise NeoFormatError(
                f"{self.path}: {lump.name} length {lump.length} is not divisible "
                f"by record size {lump.record_size}"
            )

    def lump_data(self, index: int) -> memoryview:
        lump = self.lumps[index]
        return memoryview(self.data)[lump.offset : lump.offset + lump.length]

    def vertices(self) -> list[tuple[float, float, float]]:
        return list(struct.iter_unpack("<3f", self.lump_data(0)))

    def texcoords(self) -> list[tuple[float, float]]:
        return list(struct.iter_unpack("<2f", self.lump_data(10)))

    def indices(self) -> list[int]:
        return [value[0] for value in struct.iter_unpack("<I", self.lump_data(6))]

    def draw_commands(self) -> list[DrawCommand]:
        return [DrawCommand(*values) for values in struct.iter_unpack("<IIi", self.lump_data(8))]

    def texinfo(self) -> list[TexInfo]:
        return [TexInfo(*values) for values in struct.iter_unpack("<Iii", self.lump_data(4))]

    def meshes(self) -> list[tuple[int, ...]]:
        return list(struct.iter_unpack("<36i", self.lump_data(5)))

    def lights(self) -> list[NeoLight]:
        """Decode the 84-byte renderer-light records from lump 9."""
        return [
            NeoLight(
                index,
                values[0:3], values[3], values[4:8], values[8:12],
                values[12:16], values[16:21],
            )
            for index, values in enumerate(struct.iter_unpack("<21f", self.lump_data(9)))
        ]

    def texture_info(self) -> list[TextureInfo]:
        blob = self.lump_data(2)
        if len(blob) < 4:
            return []
        count = struct.unpack_from("<I", blob, 0)[0]
        table_end = 4 + count * 4
        if count > 100_000 or table_end > len(blob):
            raise NeoFormatError("invalid embedded texture offset table")
        offsets = list(struct.unpack_from(f"<{count}I", blob, 4)) if count else []
        result: list[TextureInfo] = []
        for index, offset in enumerate(offsets):
            end = offsets[index + 1] if index + 1 < count else len(blob)
            if offset + 48 > len(blob) or end <= offset:
                continue
            name = safe_texture_name(bytes(blob[offset : offset + 32]), f"texture_{index:04d}")
            width, height, pixel_size, texture_format = struct.unpack_from("<IIII", blob, offset + 32)
            payload = bytes(blob[offset + 48 : end])
            required = width * height * pixel_size
            reference = payload.split(b"\0", 1)[0].decode("latin-1", errors="replace").strip()
            external_reference = None
            if reference.lower().endswith((".dds", ".tga", ".png")):
                external_reference = reference.replace("\\", "/").lstrip("/")
            result.append(TextureInfo(
                index, name, width, height, pixel_size, texture_format, offset, end,
                external_reference,
            ))
        return result

    def fx_shader_names(self) -> list[str]:
        blob = self.lump_data(12)
        return [
            bytes(blob[pos : pos + 132]).split(b"\0", 1)[0].decode("latin-1", errors="replace")
            for pos in range(0, len(blob), 132)
        ]

    def report(self) -> dict:
        return {
            "path": str(self.path),
            "file_size": len(self.data),
            "version": self.version,
            "supported_version": self.version in SUPPORTED_VERSIONS,
            "lumps": [asdict(lump) | {"count": lump.count} for lump in self.lumps],
        }


def resolve_mesh_material(
    neo: NeoFile,
    mesh: tuple[int, ...],
    texinfos: Sequence[TexInfo],
) -> tuple[TexInfo | None, int | None, int | None, str | None]:
    """Resolve the default material for a mesh.

    Optimized NEO shaders can bind several diffuse maps to one draw.  The
    per-vertex selector for those maps is handled separately by
    ``resolve_mesh_diffuse_slots``/``selected_diffuse_texture``.
    """
    info = texinfos[mesh[3]] if 0 <= mesh[3] < len(texinfos) else None
    texture_index = info.texture_index if info and info.texture_index >= 0 else None
    slots, shader_id, shader_name = resolve_mesh_diffuse_slots(neo, mesh, texinfos)
    if slots:
        texture_index = slots[min(slots)]
    # Some shader-driven surfaces have no direct diffuse texture and inherit
    # runtime state that is not serialized in texinfo. Their paired lightmap is
    # still self-contained and has a matching UV stream, so use it as a visible
    # fallback instead of emitting an undefined/missing material.
    if texture_index is None and 0 <= mesh[4] < len(texinfos):
        secondary = texinfos[mesh[4]]
        if secondary.texture_index >= 0:
            info = secondary
            texture_index = secondary.texture_index
    return info, texture_index, shader_id, shader_name


def resolve_mesh_diffuse_slots(
    neo: NeoFile,
    mesh: tuple[int, ...],
    texinfos: Sequence[TexInfo],
) -> tuple[dict[int, int], int | None, str | None]:
    """Return ``diffuseMapN`` texture bindings for one mesh attribute.

    Reverse engineering ``loadAttributes`` established that the first three
    uint32 values in an 80-byte serialized attribute are shader id, property
    start, and property count (in that order).
    """
    slots: dict[int, int] = {}
    explicit_slots: dict[int, int] = {}
    info = texinfos[mesh[3]] if 0 <= mesh[3] < len(texinfos) else None
    if info and info.texture_index >= 0:
        slots[0] = info.texture_index
    shader_id = None
    shader_name = None
    attributes = neo.lump_data(11)
    attribute_index = mesh[35]
    if 0 <= attribute_index < len(attributes) // 80:
        shader_id, property_start, property_count = struct.unpack_from(
            "<III", attributes, attribute_index * 80
        )
        shader_names = neo.fx_shader_names()
        if 0 <= shader_id < len(shader_names):
            shader_name = shader_names[shader_id]
        properties = neo.lump_data(13)
        if len(properties) >= 4:
            serialized_count = struct.unpack_from("<I", properties, 0)[0]
            table_end = 4 + serialized_count * 4
            if serialized_count <= 100_000 and table_end <= len(properties):
                offsets = struct.unpack_from(f"<{serialized_count}I", properties, 4)
                for property_index in range(
                        property_start, min(property_start + property_count, serialized_count)):
                    offset = offsets[property_index]
                    end = offsets[property_index + 1] if property_index + 1 < serialized_count else len(properties)
                    if offset < table_end or end > len(properties) or end - offset < 136:
                        continue
                    # Explicit effect properties are the sampler bindings used
                    # by the renderer and override texinfo's fallback texture.
                    name = bytes(properties[offset:offset + 128]).split(b"\0", 1)[0]
                    name = name.decode("latin-1", errors="replace").lower()
                    match = re.fullmatch(r"diffusemap(\d*)", name)
                    water_normal = bool(
                        shader_name and "water" in shader_name.lower() and name == "normalmap"
                    )
                    if match or name in {"diffusetexture", "basetexture"} or water_normal:
                        candidate = struct.unpack_from("<I", properties, offset + 132)[0]
                        if candidate != 0xFFFFFFFF:
                            slot = int(match.group(1) or 0) if match else 0
                            explicit_slots[slot] = candidate
    # Texinfo is only a renderer-state fallback.  When an effect serialized its
    # sampler bindings explicitly, retaining the fallback as an extra slot can
    # make layered shaders select an unrelated texture (notably effects whose
    # first real sampler is diffuseMap1 rather than diffuseMap0).
    if explicit_slots:
        slots = explicit_slots
    return slots, shader_id, shader_name


def diffuse_selector_start(
    mesh: tuple[int, ...], info: TexInfo | None, texinfos: Sequence[TexInfo],
    slots: dict[int, int],
) -> int | None:
    """Locate the packed TexCoord0.z selector stream used by OPT shaders."""
    if info is None or len(slots) < 2 or not (0 <= mesh[4] < len(texinfos)):
        return None
    following = texinfos[mesh[4]].uv_start
    span = following - info.uv_start
    # TexCoord0 is a padded float4 stored as interleaved vec2 pairs per
    # vertex: xy, then z/padding. TexCoord1 (the lightmap UV) follows it.
    if info.flags != 2 or span <= 0 or span % 2:
        return None
    return info.uv_start + 1


def mesh_texcoord_indices(
    info: TexInfo, selector_start: int | None, local_vertex: int,
) -> tuple[int, int | None]:
    """Return UV and optional selector indices for a mesh-local vertex."""
    stride = 2 if selector_start is not None else 1
    uv_index = info.uv_start + local_vertex * stride
    selector_index = selector_start + local_vertex * stride if selector_start is not None else None
    return uv_index, selector_index


def selected_diffuse_texture(
    slots: dict[int, int], texcoords: Sequence[tuple[float, float]],
    selector_indices: Sequence[int | None],
) -> int | None:
    """Choose a batched diffuse texture from a triangle's TexCoord0.z values."""
    fallback = slots[min(slots)] if slots else None
    values = sorted(
        texcoords[index][0] for index in selector_indices
        if index is not None and 0 <= index < len(texcoords)
        and math.isfinite(texcoords[index][0])
    )
    if not values:
        return fallback
    slot = int(round(values[len(values) // 2]))
    return slots.get(slot, fallback)


def triangles_for_primitive(mode: int, values: Sequence) -> Iterable[tuple]:
    """Triangulate one OpenGL primitive. Input indices remain zero-based."""
    if mode == GL_TRIANGLES:
        for pos in range(0, len(values) - 2, 3):
            yield values[pos], values[pos + 1], values[pos + 2]
    elif mode == GL_TRIANGLE_STRIP:
        for pos in range(len(values) - 2):
            if pos & 1:
                tri = (values[pos + 1], values[pos], values[pos + 2])
            else:
                tri = (values[pos], values[pos + 1], values[pos + 2])
            yield tri
    elif mode in (GL_TRIANGLE_FAN, GL_POLYGON):
        if len(values) >= 3:
            anchor = values[0]
            for pos in range(1, len(values) - 1):
                yield anchor, values[pos], values[pos + 1]
    elif mode == GL_QUADS:
        for pos in range(0, len(values) - 3, 4):
            a, b, c, d = values[pos : pos + 4]
            yield a, b, c
            yield a, c, d
    elif mode == GL_QUAD_STRIP:
        for pos in range(0, len(values) - 3, 2):
            a, b, c, d = values[pos : pos + 4]
            yield a, b, d
            yield a, d, c


def export_obj(
    neo: NeoFile,
    destination: Path,
    scale: float,
    source_axes: bool,
    blender_axes: bool,
    flip_v: bool,
    split_objects: bool,
    override: bool = False,
) -> dict:
    vertices = neo.vertices()
    texcoords = neo.texcoords()
    indices = neo.indices()
    commands = neo.draw_commands()
    mesh_records = neo.meshes()
    # Mesh records may be material-sorted rather than draw-sorted. Field 2 is
    # the explicit draw-command ID; pairing both lumps by list position only
    # works accidentally for maps whose records are already in ID order.
    meshes_by_draw_id = {record[2]: record for record in mesh_records}
    texinfo_records = neo.texinfo()
    textures = {item.index: item for item in neo.texture_info()}
    destination.parent.mkdir(parents=True, exist_ok=True)

    written_faces = 0
    skipped_faces = 0
    unsupported_modes: set[int] = set()
    luminance_alpha_textures: set[int] = set()
    uniform_alpha_textures: set[int] = set()

    with destination.open("w", encoding="utf-8", newline="\n") as obj:
        obj.write(f"# Converted from {neo.path.name}\n")
        obj.write(f"# NEO version {neo.version}\n")
        obj.write(f"mtllib {destination.stem}.mtl\n")
        obj.write("o neo_world\n")

        for x, y, z in vertices:
            if not all(math.isfinite(value) for value in (x, y, z)):
                x = y = z = 0.0
            if source_axes:
                obj.write(f"v {x * scale:.9g} {z * scale:.9g} {-y * scale:.9g}\n")
            elif blender_axes:
                # Source/GoldSrc Y points left; mirror it for Blender's visual
                # convention. NEO's stored strip winding is opposite Blender's,
                # so the handedness change also corrects the visible face side.
                # Then rotate -90 degrees around Blender X so the map has the
                # expected upright orientation in Blender.
                # Finally mirror Blender X to match the map's intended left/
                # right orientation. This restores handedness, so OBJ face
                # winding is reversed below to keep surfaces outward-facing.
                obj.write(f"v {-x * scale:.9g} {z * scale:.9g} {y * scale:.9g}\n")
            else:
                obj.write(f"v {x * scale:.9g} {y * scale:.9g} {z * scale:.9g}\n")

        # OBJ permits a UV pool independent from the vertex pool. Blender uses
        # the per-face v/vt pairs written below, so NEO's separate streams map
        # naturally to this representation.
        for u, v in texcoords:
            if not math.isfinite(u) or not math.isfinite(v):
                u = v = 0.0
            output_v = 1.0 - v if flip_v else v
            obj.write(f"vt {u:.9g} {output_v:.9g}\n")

        for number, command in enumerate(commands):
            label = f"draw_command_{number:05d}"
            obj.write(f"{'o' if split_objects else 'g'} {label}\n")
            info: TexInfo | None = None
            uv_start: int | None = None
            texture_index: int | None = None
            active_material: str | None = None
            mesh_record = meshes_by_draw_id.get(number)
            diffuse_slots: dict[int, int] = {}
            selector_start: int | None = None
            if mesh_record is not None:
                info, texture_index, _, shader_name = resolve_mesh_material(
                    neo, mesh_record, texinfo_records
                )
                diffuse_slots, _, _ = resolve_mesh_diffuse_slots(
                    neo, mesh_record, texinfo_records
                )
                if shader_name:
                    lowered_shader = shader_name.lower()
                    if "additive" in lowered_shader or "billboard" in lowered_shader:
                        # NEO billboards commonly store an opaque black
                        # background and rely on shader blending, just like its
                        # additive effects (for example the ``helth`` marker).
                        luminance_alpha_textures.update(diffuse_slots.values())
                    elif "water" in lowered_shader or "translucent" in lowered_shader:
                        uniform_alpha_textures.update(diffuse_slots.values())
                if info is not None:
                    uv_start = info.uv_start
                    selector_start = diffuse_selector_start(
                        mesh_record, info, texinfo_records, diffuse_slots
                    )
            if command.indexed:
                # Indexed commands point at a small header in the index lump:
                #   uint32 base_vertex, uint32 vertex_count, uint32 indices[]
                # Indices are relative to base_vertex and 0xffff is a primitive
                # restart marker.  The signed draw count excludes the header.
                if command.start + 2 > len(indices):
                    continue
                base_vertex = indices[command.start]
                first = command.start + 2
                values = indices[first : min(first + command.count, len(indices))]
                primitive_runs: list[list[tuple[int, int | None]]] = []
                run: list[tuple[int, int | None]] = []
                for value in values:
                    if value in (0xFFFF, 0xFFFFFFFF):
                        if run:
                            primitive_runs.append(run)
                            run = []
                    else:
                        if info is not None:
                            uv_index, selector_index = mesh_texcoord_indices(
                                info, selector_start, value
                            )
                        else:
                            uv_index = selector_index = None
                        run.append((base_vertex + value, uv_index, selector_index))
                if run:
                    primitive_runs.append(run)
            else:
                end = command.start + command.count
                primitive_runs = [[
                    (vertex_index, *mesh_texcoord_indices(info, selector_start, local))
                    for local, vertex_index in enumerate(range(command.start, min(end, len(vertices))))
                ]] if info is not None else []

            if command.mode not in {
                GL_TRIANGLES, GL_TRIANGLE_STRIP, GL_TRIANGLE_FAN,
                GL_QUADS, GL_QUAD_STRIP, GL_POLYGON,
            }:
                unsupported_modes.add(command.mode)
                continue

            for values in primitive_runs:
                for triangle in triangles_for_primitive(command.mode, values):
                    if blender_axes:
                        triangle = (triangle[0], triangle[2], triangle[1])
                    vertex_indices = tuple(item[0] for item in triangle)
                    uv_indices = tuple(item[1] for item in triangle)
                    if len(set(vertex_indices)) != 3 or any(index >= len(vertices) for index in vertex_indices):
                        skipped_faces += 1
                        continue
                    if all(index is not None and 0 <= index < len(texcoords) for index in uv_indices):
                        corners = [
                            f"{vertex_index + 1}/{uv_index + 1}"
                            for vertex_index, uv_index in zip(vertex_indices, uv_indices)
                        ]
                    else:
                        corners = [str(index + 1) for index in vertex_indices]
                    face_texture = selected_diffuse_texture(
                        diffuse_slots, texcoords, [item[2] for item in triangle]
                    )
                    if face_texture is None:
                        face_texture = texture_index
                    if face_texture is not None:
                        texture = textures.get(face_texture)
                        suffix = texture.name if texture else f"texture_{face_texture:04d}"
                        face_material = f"neo_{face_texture:04d}_{suffix}"
                        if face_material != active_material:
                            obj.write(f"usemtl {face_material}\n")
                            active_material = face_material
                    obj.write("f " + " ".join(corners) + "\n")
                    written_faces += 1

    alpha_textures = {
        index for index, texture in textures.items()
        if texture_has_alpha(neo, texture)
    }
    write_mtl(
        destination.with_suffix(".mtl"), textures, alpha_textures,
        luminance_alpha_textures, uniform_alpha_textures, neo, override,
    )

    return {
        "output": str(destination),
        "vertices": len(vertices),
        "draw_commands": len(commands),
        "texcoords": len(texcoords),
        "materials": len(textures),
        "faces": written_faces,
        "skipped_faces": skipped_faces,
        "unsupported_primitive_modes": sorted(unsupported_modes),
        "alpha_materials": len(
            alpha_textures | luminance_alpha_textures | uniform_alpha_textures
        ),
        "shader_luminance_alpha_materials": len(luminance_alpha_textures),
    }


def texture_has_alpha(neo: NeoFile, texture: TextureInfo) -> bool:
    """Return whether a texture stores a usable alpha channel."""
    # NEO lightmap alpha stores auxiliary baked-lighting information. It is
    # not surface opacity, including when a missing diffuse texture causes the
    # lightmap to be exported as a visible fallback.
    if texture.name.lower().startswith("lightmap"):
        return False
    if texture.external_reference:
        source = neo.path.parent.parent / "tex" / Path(texture.external_reference)
        try:
            data = source.read_bytes()
        except OSError:
            return False
        header = data[:128]
        suffix = source.suffix.lower()
        if suffix == ".dds" and len(header) >= 108 and header[:4] == b"DDS ":
            pixel_flags = struct.unpack_from("<I", header, 80)[0]
            fourcc = header[84:88]
            alpha_mask = struct.unpack_from("<I", header, 104)[0]
            if pixel_flags & 0x1 or alpha_mask or fourcc in {b"DXT3", b"DXT5"}:
                return True
            if fourcc == b"DXT1":
                width = struct.unpack_from("<I", header, 16)[0]
                height = struct.unpack_from("<I", header, 12)[0]
                block_count = ((width + 3) // 4) * ((height + 3) // 4)
                for offset in range(128, min(128 + block_count * 8, len(data)), 8):
                    if offset + 8 > len(data):
                        break
                    color0, color1, selectors = struct.unpack_from("<HHI", data, offset)
                    # DXT1 enables transparent palette entry 3 only when
                    # color0 <= color1 and at least one pixel selects entry 3.
                    if color0 <= color1 and any(
                            (selectors >> shift) & 0x3 == 3 for shift in range(0, 32, 2)):
                        return True
            return False
        if suffix == ".tga" and len(header) >= 18:
            return header[16] == 32 and bool(header[17] & 0x0F)
        if suffix == ".png" and len(header) >= 26 and header[:8] == b"\x89PNG\r\n\x1a\n":
            return header[25] in {4, 6}
        return False

    if texture.pixel_size != 4:
        return False
    blob = neo.lump_data(2)
    pixel_start = texture.offset + 48
    pixel_end = min(pixel_start + texture.width * texture.height * 4, texture.end)
    if pixel_end <= pixel_start:
        return False
    return any(alpha < 255 for alpha in blob[pixel_start + 3:pixel_end:4])


def write_mtl(
    path: Path, textures: dict[int, TextureInfo], alpha_textures: set[int] | None = None,
    luminance_alpha_textures: set[int] | None = None,
    uniform_alpha_textures: set[int] | None = None,
    neo: NeoFile | None = None,
    override: bool = False,
) -> None:
    alpha_textures = alpha_textures or set()
    luminance_alpha_textures = luminance_alpha_textures or set()
    uniform_alpha_textures = uniform_alpha_textures or set()
    additive_images: dict[int, str] = {}
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg and neo is not None:
        texture_dir = path.parent / "textures"
        texture_dir.mkdir(parents=True, exist_ok=True)
        # Shader blend semantics take precedence over a DDS alpha declaration.
        # Several NEO effect textures are encoded as RGBA but contain 255 in
        # every alpha pixel; the runtime still makes their black background
        # transparent through additive/billboard blending.
        for index in luminance_alpha_textures:
            texture = textures.get(index)
            if texture is None or not texture.external_reference:
                continue
            source = neo.path.parent.parent / "tex" / Path(texture.external_reference)
            output_name = f"{index:04d}_{texture.name}_additive.tga"
            output = texture_dir / output_name
            if output.is_file() and not override:
                additive_images[index] = f"textures/{output_name}"
                continue
            if not source.is_file():
                continue
            try:
                result = subprocess.run(
                    [ffmpeg, "-y", "-loglevel", "error", "-i", str(source),
                     "-vf", "colorkey=0x000000:0.12:0.08,format=rgba", str(output)],
                    text=True, capture_output=True, check=False, timeout=30,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            if result.returncode == 0 and output.is_file():
                additive_images[index] = f"textures/{output_name}"
    with path.open("w", encoding="utf-8", newline="\n") as mtl:
        mtl.write("# Materials recovered from the NEO embedded texture directory\n")
        for index, texture in sorted(textures.items()):
            material = f"neo_{index:04d}_{texture.name}"
            extension = Path(texture.external_reference).suffix if texture.external_reference else ".tga"
            image = f"textures/{index:04d}_{texture.name}{extension.lower()}"
            mtl.write(f"\nnewmtl {material}\n")
            mtl.write("Ka 0.2 0.2 0.2\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\n")
            opacity_image = additive_images.get(index)
            dissolve = 0.45 if index in uniform_alpha_textures and index not in alpha_textures else 1.0
            mtl.write(f"d {dissolve:.2f}\nillum 1\n")
            mtl.write(f"map_Kd {opacity_image or image}\n")
            if index in luminance_alpha_textures:
                # Additive NEO shaders commonly store glowing pixels on an
                # opaque black background. Use image luminance as the closest
                # portable OBJ opacity approximation to additive blending.
                mtl.write(f"map_d {opacity_image}\n" if opacity_image else f"map_d -imfchan r {image}\n")
            elif index in alpha_textures:
                # Blender's OBJ importer uses map_d to connect the image alpha
                # channel and select a transparent material blend mode.
                mtl.write(f"map_d {image}\n")


def indexed_runs(command: DrawCommand, indices: Sequence[int]) -> tuple[int, list[list[int]]]:
    """Return the base vertex and restart-separated local-index runs."""
    if command.start + 2 > len(indices):
        return 0, []
    base_vertex = indices[command.start]
    values = indices[command.start + 2 : min(command.start + 2 + command.count, len(indices))]
    runs: list[list[int]] = []
    run: list[int] = []
    for value in values:
        if value in (0xFFFF, 0xFFFFFFFF):
            if run:
                runs.append(run)
                run = []
        else:
            run.append(value)
    if run:
        runs.append(run)
    return base_vertex, runs


def export_hammer(
    neo: NeoFile, destination: Path, scale: float, flip_v: bool,
    export_all_textures: bool = False,
    vmf_name: str | None = None,
    override: bool = False,
    target_game: str = "gmod",
    sky_name: str = "sky_day01_01",
) -> dict:
    """Write an initial Source SDK model/material/VMF conversion scaffold."""
    vertices = neo.vertices()
    texcoords = neo.texcoords()
    indices = neo.indices()
    commands = neo.draw_commands()
    texinfos = neo.texinfo()
    meshes = {record[2]: record for record in neo.meshes()}
    textures = {item.index: item for item in neo.texture_info()}
    resolved_by_draw: dict[int, tuple[TexInfo | None, int | None, int | None, str | None]] = {}
    diffuse_slots_by_draw: dict[int, dict[int, int]] = {}
    for draw_id, record in meshes.items():
        resolved = resolve_mesh_material(neo, record, texinfos)
        resolved_by_draw[draw_id] = resolved
        diffuse_slots_by_draw[draw_id] = resolve_mesh_diffuse_slots(neo, record, texinfos)[0]
    used_diffuse_textures = {
        texture_index for slots in diffuse_slots_by_draw.values()
        for texture_index in slots.values()
    }
    map_name = neo.path.stem
    model_rel = f"neo_maps/{map_name}"
    material_rel = f"models/{model_rel}"
    modelsrc = destination / "modelsrc" / model_rel
    materials = destination / "materials" / material_rel
    brush_materials_dir = destination / "materials" / model_rel
    materialsrc = destination / "materialsrc" / material_rel
    modelsrc.mkdir(parents=True, exist_ok=True)
    materials.mkdir(parents=True, exist_ok=True)
    brush_materials_dir.mkdir(parents=True, exist_ok=True)
    materialsrc.mkdir(parents=True, exist_ok=True)

    texture_outputs = extract_textures(neo, destination.parent / "textures", override)
    extracted = {item["index"]: Path(item["output"]) for item in texture_outputs if "output" in item}
    special_markers = ("alpha", "trans", "sky", "water", "billboard", "_bb", "scroll", "anim")
    manifest: list[dict] = []
    entities: list[str] = []
    next_entity_id = 2
    logical_vertex_count = neo.lumps[1].count or len(vertices)
    position_vertices = vertices[: min(logical_vertex_count, len(vertices))]
    raw_min = [min(point[axis] for point in position_vertices) for axis in range(3)]
    raw_max = [max(point[axis] for point in position_vertices) for axis in range(3)]
    hammer_center = [
        (raw_min[axis] + raw_max[axis]) * 0.5 * scale for axis in range(3)
    ]
    bounds_min = [raw_min[axis] * scale - hammer_center[axis] for axis in range(3)]
    bounds_max = [raw_max[axis] * scale - hammer_center[axis] for axis in range(3)]

    for number, command in enumerate(commands):
        mesh = meshes.get(number)
        if mesh is None or mesh[3] < 0 or mesh[3] >= len(texinfos):
            continue
        info, texture_index, shader_id, shader_name = resolved_by_draw[number]
        if info is None or texture_index is None:
            continue
        material_fallback = texinfos[mesh[3]].texture_index < 0
        texture = textures.get(texture_index)
        texture_name = texture.name if texture else f"texture_{texture_index:04d}"
        material = f"neo_{texture_index:04d}_{texture_name}"
        diffuse_slots = diffuse_slots_by_draw[number]
        selector_start = diffuse_selector_start(mesh, info, texinfos, diffuse_slots)
        special = shader_name is not None or len(
            [value for value in mesh[3:35] if value >= 0]
        ) > 2 or any(
            marker in texture_name.lower() for marker in special_markers
        )

        if command.indexed:
            base_vertex, raw_runs = indexed_runs(command, indices)
            runs = [[
                (base_vertex + value, *mesh_texcoord_indices(info, selector_start, value))
                for value in run
            ] for run in raw_runs]
        else:
            end = min(command.start + command.count, len(vertices))
            runs = [[(vertex_index, *mesh_texcoord_indices(info, selector_start, local))
                     for local, vertex_index in enumerate(range(command.start, end))]]

        triangles: list[tuple] = []
        for run in runs:
            for triangle in triangles_for_primitive(command.mode, run):
                # NEO's native strip front-face convention is opposite Source
                # SMD/MDL winding. Reverse only the Hammer model path; the
                # Blender-axis OBJ mirror already changes handedness itself.
                triangle = (triangle[0], triangle[2], triangle[1])
                vertex_ids = [corner[0] for corner in triangle]
                uv_ids = [corner[1] for corner in triangle]
                if len(set(vertex_ids)) != 3:
                    continue
                if any(v < 0 or v >= len(vertices) for v in vertex_ids):
                    continue
                if any(t < 0 or t >= len(texcoords) for t in uv_ids):
                    continue
                triangles.append(triangle)
        if not triangles:
            continue

        stem = f"draw_command_{number:05d}"
        smd_path = modelsrc / f"{stem}.smd"
        with smd_path.open("w", encoding="utf-8", newline="\n") as smd:
            smd.write('version 1\nnodes\n0 "root" -1\nend\nskeleton\ntime 0\n')
            smd.write("0 0 0 0 0 0 0\nend\ntriangles\n")
            for triangle in triangles:
                face_texture_index = selected_diffuse_texture(
                    diffuse_slots, texcoords, [corner[2] for corner in triangle]
                )
                if face_texture_index is None:
                    face_texture_index = texture_index
                face_texture = textures.get(face_texture_index)
                face_texture_name = (
                    face_texture.name if face_texture else f"texture_{face_texture_index:04d}"
                )
                face_material = f"neo_{face_texture_index:04d}_{face_texture_name}"
                points = [vertices[corner[0]] for corner in triangle]
                for point in points:
                    for axis in range(3):
                        coordinate = point[axis] * scale
                        coordinate -= hammer_center[axis]
                        bounds_min[axis] = min(bounds_min[axis], coordinate)
                        bounds_max[axis] = max(bounds_max[axis], coordinate)
                ab = tuple(points[1][i] - points[0][i] for i in range(3))
                ac = tuple(points[2][i] - points[0][i] for i in range(3))
                normal = (
                    ab[1] * ac[2] - ab[2] * ac[1],
                    ab[2] * ac[0] - ab[0] * ac[2],
                    ab[0] * ac[1] - ab[1] * ac[0],
                )
                length = math.sqrt(sum(value * value for value in normal))
                if length <= 1e-12:
                    continue
                normal = tuple(value / length for value in normal)
                smd.write(face_material + "\n")
                for corner, point in zip(triangle, points):
                    u, v = texcoords[corner[1]]
                    if flip_v:
                        v = 1.0 - v
                    smd.write(
                        f"0 {point[0] * scale - hammer_center[0]:.9g} "
                        f"{point[1] * scale - hammer_center[1]:.9g} "
                        f"{point[2] * scale - hammer_center[2]:.9g} "
                        f"{normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g} {u:.9g} {v:.9g}\n"
                    )
            smd.write("end\n")

        qc_path = modelsrc / f"{stem}.qc"
        qc_path.write_text(
            f'$modelname "{model_rel}/{stem}.mdl"\n'
            f'$body "body" "{stem}.smd"\n'
            f'$cdmaterials "{material_rel}/"\n'
            '$staticprop\n$surfaceprop "concrete"\n'
            f'$sequence "idle" "{stem}.smd" fps 1\n',
            encoding="utf-8",
        )
        entities.append(
            f'entity\n{{\n"id" "{next_entity_id}"\n"classname" "prop_static"\n'
            f'"model" "models/{model_rel}/{stem}.mdl"\n"origin" "0 0 0"\n'
            '"angles" "0 0 0"\n"solid" "6"\n}\n'
        )
        next_entity_id += 1
        manifest.append({
            "draw_command": number, "material": material, "texture": texture_name,
            "diffuse_slots": diffuse_slots,
            "triangles": len(triangles), "special_shader_candidate": special,
            "neo_shader_id": shader_id,
            "neo_shader": shader_name,
            "material_fallback": "lightmap" if material_fallback else None,
            "smd": str(smd_path), "qc": str(qc_path),
        })

    for index, texture in textures.items():
        if not export_all_textures and index not in used_diffuse_textures:
            continue
        material = f"neo_{index:04d}_{texture.name}"
        vmt = materials / f"{material}.vmt"
        vmt.write_text(
            '"VertexLitGeneric"\n{\n'
            f'    "$basetexture" "{material_rel}/{material}"\n'
            '}\n', encoding="utf-8"
        )
        if export_all_textures:
            # Keep world-brush aliases outside materials/models. Hammer treats
            # that tree as model-only and may hide or fail to preview its VMTs.
            (brush_materials_dir / f"{material}_brush.vmt").write_text(
                '"LightmappedGeneric"\n{\n'
                f'    "$basetexture" "{material_rel}/{material}"\n'
                '}\n', encoding="utf-8"
            )
        source = extracted.get(index)
        if source and source.is_file():
            shutil.copyfile(source, materialsrc / f"{material}{source.suffix.lower()}")

    def box_solid(
        solid_id: int,
        side_id: int,
        low: tuple,
        high: tuple,
        visible_side: int | None = None,
    ) -> tuple[str, int]:
        x1, y1, z1 = low
        x2, y2, z2 = high
        planes = [
            ((x1, y1, z2), (x1, y2, z2), (x2, y2, z2)),
            ((x1, y2, z1), (x1, y1, z1), (x2, y1, z1)),
            ((x1, y1, z1), (x1, y2, z1), (x1, y2, z2)),
            ((x2, y2, z1), (x2, y1, z1), (x2, y1, z2)),
            ((x2, y1, z1), (x1, y1, z1), (x1, y1, z2)),
            ((x1, y2, z1), (x2, y2, z1), (x2, y2, z2)),
        ]
        text_parts = [f'solid\n{{\n"id" "{solid_id}"\n']
        for plane_number, plane in enumerate(planes):
            formatted = " ".join(
                f"({point[0]:.9g} {point[1]:.9g} {point[2]:.9g})" for point in plane
            )
            material_name = (
                "TOOLS/TOOLSSKYBOX" if plane_number == visible_side else "TOOLS/TOOLSNODRAW"
            )
            text_parts.append(
                f'side\n{{\n"id" "{side_id}"\n"plane" "{formatted}"\n'
                f'"material" "{material_name}"\n'
                '"uaxis" "[1 0 0 0] 0.25"\n"vaxis" "[0 -1 0 0] 0.25"\n'
                '"rotation" "0"\n"lightmapscale" "16"\n"smoothing_groups" "0"\n}\n'
            )
            side_id += 1
        text_parts.append("}\n")
        return "".join(text_parts), side_id

    world_solids = ""
    if all(math.isfinite(value) for value in bounds_min + bounds_max):
        margin = 256.0
        thickness = 64.0
        x1, y1, z1 = (math.floor(value - margin) for value in bounds_min)
        x2, y2, z2 = (math.ceil(value + margin) for value in bounds_max)
        slabs = [
            ((x1 - thickness, y1 - thickness, z1 - thickness), (x2 + thickness, y2 + thickness, z1)),
            ((x1 - thickness, y1 - thickness, z2), (x2 + thickness, y2 + thickness, z2 + thickness)),
            ((x1 - thickness, y1, z1), (x1, y2, z2)),
            ((x2, y1, z1), (x2 + thickness, y2, z2)),
            ((x1, y1 - thickness, z1), (x2, y1, z2)),
            ((x1, y2, z1), (x2, y2 + thickness, z2)),
        ]
        solid_id = next_entity_id
        side_id = solid_id + len(slabs)
        for slab_number, (low, high) in enumerate(slabs):
            # Keep one real world face so vbsp emits surfedges. The inward
            # face of the ceiling becomes sky; every structural face remains nodraw.
            visible_side = 1 if slab_number == 1 else None
            solid, side_id = box_solid(solid_id, side_id, low, high, visible_side)
            world_solids += solid
            solid_id += 1
        next_entity_id = side_id
        spawn = (
            (bounds_min[0] + bounds_max[0]) * 0.5,
            (bounds_min[1] + bounds_max[1]) * 0.5,
            bounds_min[2] + 64.0,
        )
        if target_game == "css":
            # Keep the two team spawns close to the model-map origin while
            # leaving enough room that the player hulls do not overlap.
            for entity_id, classname, x_offset in (
                (next_entity_id, "info_player_terrorist", -32.0),
                (next_entity_id + 1, "info_player_counterterrorist", 32.0),
            ):
                entities.append(
                    f'entity\n{{\n"id" "{entity_id}"\n"classname" "{classname}"\n'
                    f'"origin" "{spawn[0] + x_offset:.9g} {spawn[1]:.9g} {spawn[2]:.9g}"\n'
                    '"angles" "0 0 0"\n}\n'
                )
        else:
            entities.append(
                f'entity\n{{\n"id" "{next_entity_id}"\n"classname" "info_player_start"\n'
                f'"origin" "{spawn[0]:.9g} {spawn[1]:.9g} {spawn[2]:.9g}"\n'
                '"angles" "0 0 0"\n}\n'
            )

    vmf = destination / f"{vmf_name or map_name}.vmf"
    vmf.write_text(
        'versioninfo\n{\n"editorversion" "400"\n"editorbuild" "0"\n'
        '"mapversion" "1"\n"formatversion" "100"\n"prefab" "0"\n}\n'
        'visgroups\n{\n}\nviewsettings\n{\n"bSnapToGrid" "1"\n"bShowGrid" "1"\n'
        '"bShowLogicalGrid" "0"\n"nGridSpacing" "64"\n}\n'
        'world\n{\n"id" "1"\n"mapversion" "1"\n"classname" "worldspawn"\n'
        f'"skyname" "{sky_name}"\n' + world_solids + '}\n'
        + "".join(entities) + 'cameras\n{\n"activecamera" "-1"\n}\ncordons\n{\n"active" "0"\n}\n',
        encoding="utf-8",
    )
    manifest_path = destination / "hammer_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (destination / "README_HAMMER.txt").write_text(
        "Initial Source/Hammer conversion\n\n"
        "1. Convert every image under materialsrc/ to VTF with Valve's vtex.\n"
        "2. Copy the generated VTF files and the matching VMT files under materials/ "
        "into your game's materials/ directory, preserving the models/neo_maps/... path.\n"
        "3. Compile each QC under modelsrc/ with the Source branch's studiomdl using -game.\n"
        "4. Open the generated VMF in Hammer after the compiled MDLs are under "
        "the game's models/neo_maps/... directory.\n\n"
        "The manifest flags likely animated, translucent, billboard, and other special "
        "materials. Those are placeholders in this initial static conversion.\n",
        encoding="utf-8",
    )
    return {
        "vmf": str(vmf), "models": len(manifest), "manifest": str(manifest_path),
        "source_translation": [-value for value in hammer_center],
        "bounds": {"min": bounds_min, "max": bounds_max},
    }


MAP_FACE_RE = re.compile(
    r"^\s*\(\s*([^)]*?)\s*\)\s*\(\s*([^)]*?)\s*\)\s*"
    r"\(\s*([^)]*?)\s*\)\s+(\S+)"
)


def _vsub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(a[i] - b[i] for i in range(3))


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum(a[i] * b[i] for i in range(3))


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _plane(points: Sequence[tuple[float, float, float]]) -> tuple[tuple[float, float, float], float] | None:
    normal = _cross(_vsub(points[1], points[0]), _vsub(points[2], points[0]))
    length = math.sqrt(_dot(normal, normal))
    if length < 1e-8:
        return None
    normal = tuple(value / length for value in normal)
    return normal, _dot(normal, points[0])


def _intersection(a, b, c) -> tuple[float, float, float] | None:
    na, da = a
    nb, db = b
    nc, dc = c
    denominator = _dot(na, _cross(nb, nc))
    if abs(denominator) < 1e-9:
        return None
    bc, ca, ab = _cross(nb, nc), _cross(nc, na), _cross(na, nb)
    return tuple((da * bc[i] + db * ca[i] + dc * ab[i]) / denominator for i in range(3))


def parse_decompiled_world_brushes(path: Path) -> list[list[dict]]:
    """Parse worldspawn brushes from a Valve 220 GoldSrc MAP file."""
    brushes: list[list[dict]] = []
    depth = 0
    in_world = False
    current: list[dict] | None = None
    for line in path.read_text(encoding="latin-1", errors="replace").splitlines():
        stripped = line.strip()
        if stripped == "{":
            depth += 1
            if depth == 1:
                in_world = not brushes
            elif in_world and depth == 2:
                current = []
            continue
        if stripped == "}":
            if in_world and depth == 2 and current is not None:
                if current:
                    brushes.append(current)
                current = None
            depth -= 1
            if depth == 0 and in_world:
                break
            continue
        if not in_world or depth != 2 or current is None:
            continue
        match = MAP_FACE_RE.match(line)
        if not match:
            continue
        try:
            points = [tuple(map(float, match.group(i).split())) for i in range(1, 4)]
        except ValueError:
            continue
        plane = _plane(points)
        if plane:
            current.append({"points": points, "texture": match.group(4), "plane": plane})
    return brushes


def parse_decompiled_entities(path: Path) -> list[dict]:
    """Parse top-level Valve MAP entities, their keyvalues, and brush planes."""
    entities: list[dict] = []
    current: dict | None = None
    current_brush: list[dict] | None = None
    depth = 0
    keyvalue_re = re.compile(r'^\s*"([^"]+)"\s+"([^"]*)"')
    for line in path.read_text(encoding="latin-1", errors="replace").splitlines():
        stripped = line.strip()
        if stripped == "{":
            depth += 1
            if depth == 1:
                current = {"keyvalues": {}, "brushes": []}
            elif depth == 2 and current is not None:
                current_brush = []
            continue
        if stripped == "}":
            if depth == 2 and current is not None and current_brush is not None:
                if current_brush:
                    current["brushes"].append(current_brush)
                current_brush = None
            elif depth == 1 and current is not None:
                entities.append(current)
                current = None
            depth -= 1
            continue
        if current is None:
            continue
        if depth == 1:
            match = keyvalue_re.match(line)
            if match:
                current["keyvalues"][match.group(1)] = match.group(2)
        elif depth == 2 and current_brush is not None:
            match = MAP_FACE_RE.match(line)
            if not match:
                continue
            try:
                points = [tuple(map(float, match.group(i).split())) for i in range(1, 4)]
            except ValueError:
                continue
            plane = _plane(points)
            if plane:
                current_brush.append({"points": points, "texture": match.group(4), "plane": plane})
    return entities


def parse_decompiled_vmf_entities(path: Path) -> list[dict]:
    """Parse J.A.C.K/Source VMF world and entity brushes into MAP-like data."""
    root = {"name": "root", "keyvalues": {}, "children": []}
    stack = [root]
    pending_name: str | None = None
    keyvalue_re = re.compile(r'^\s*"([^"]+)"\s+"([^"]*)"')
    block_name_re = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*$")

    for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        match = keyvalue_re.match(line)
        if match:
            # VMF allows repeated output keys. The NEO entity translator only
            # consumes ordinary scalar keys, so retain the last value here.
            stack[-1]["keyvalues"][match.group(1)] = match.group(2)
            continue
        if stripped == "{":
            if pending_name is None:
                raise NeoFormatError(f"VMF block without a name at {path}:{line_number}")
            child = {"name": pending_name.lower(), "keyvalues": {}, "children": []}
            stack[-1]["children"].append(child)
            stack.append(child)
            pending_name = None
            continue
        if stripped == "}":
            if len(stack) == 1:
                raise NeoFormatError(f"unexpected VMF closing brace at {path}:{line_number}")
            stack.pop()
            pending_name = None
            continue
        match = block_name_re.match(line)
        if match:
            pending_name = match.group(1)

    if len(stack) != 1:
        raise NeoFormatError(f"unclosed VMF block in {path}")

    plane_re = re.compile(r"\(([^)]*)\)\s*\(([^)]*)\)\s*\(([^)]*)\)")

    def brushes_from_node(node: dict) -> list[list[dict]]:
        brushes: list[list[dict]] = []
        for solid in (child for child in node["children"] if child["name"] == "solid"):
            brush: list[dict] = []
            for side in (child for child in solid["children"] if child["name"] == "side"):
                plane_text = side["keyvalues"].get("plane", "")
                match = plane_re.fullmatch(plane_text.strip())
                if not match:
                    continue
                try:
                    points = [tuple(map(float, match.group(index).split())) for index in range(1, 4)]
                except ValueError:
                    continue
                plane = _plane(points)
                if plane:
                    brush.append({
                        "points": points,
                        "texture": side["keyvalues"].get("material", "NULL"),
                        "plane": plane,
                        "vmf_uaxis": side["keyvalues"].get("uaxis"),
                        "vmf_vaxis": side["keyvalues"].get("vaxis"),
                    })
            if brush:
                brushes.append(brush)
        return brushes

    entities: list[dict] = []
    world = next((child for child in root["children"] if child["name"] == "world"), None)
    if world is None:
        raise NeoFormatError(f"VMF contains no world block: {path}")
    entities.append({"keyvalues": dict(world["keyvalues"]), "brushes": brushes_from_node(world)})
    for node in (child for child in root["children"] if child["name"] == "entity"):
        entities.append({"keyvalues": dict(node["keyvalues"]), "brushes": brushes_from_node(node)})
    return entities


def read_bsp_entity_lump(bsp_path: Path) -> tuple[int, bytes]:
    """Read and validate the GoldSrc BSP entity lump."""
    data = bsp_path.read_bytes()
    if len(data) < 12:
        raise NeoFormatError(f"BSP file is too small: {bsp_path}")
    version = struct.unpack_from("<i", data, 0)[0]
    entity_offset, entity_length = struct.unpack_from("<ii", data, 4)
    if (entity_offset < 0 or entity_length < 0 or
            entity_offset + entity_length > len(data)):
        raise NeoFormatError(f"invalid BSP entity lump bounds: {bsp_path}")
    return version, data[entity_offset:entity_offset + entity_length]


def read_bsp_map_metadata(bsp_path: Path) -> dict:
    """Recover reusable map-level metadata from the original BSP."""
    version, entity_lump = read_bsp_entity_lump(bsp_path)
    worldspawn: dict[str, str] = {}
    for block in re.findall(br"\{[^{}]*\}", entity_lump, re.DOTALL):
        raw_values = dict(re.findall(br'"([^"]+)"\s+"([^"]*)"', block))
        classname = raw_values.get(b"classname", b"").decode("ascii", errors="replace")
        if classname.lower() != "worldspawn":
            continue
        worldspawn = {
            key.decode("ascii", errors="replace"): value.decode("cp932", errors="replace")
            for key, value in raw_values.items()
        }
        break
    sky_name = worldspawn.get("skyname", "").strip()
    if sky_name and not re.fullmatch(r"[A-Za-z0-9_.-]+", sky_name):
        sky_name = ""
    return {
        "bsp_version": version,
        "worldspawn": worldspawn,
        "sky_name": sky_name or None,
    }


def export_bsp_location_names(bsp_path: Path, destination: Path) -> dict:
    """Recover NEO PLACE_NAME labels from a GoldSrc BSP entity lump."""
    version, entity_lump = read_bsp_entity_lump(bsp_path)

    entities: list[dict] = []
    by_id: dict[str, dict] = {}
    for block in re.findall(br"\{[^{}]*\}", entity_lump, re.DOTALL):
        raw_values = dict(re.findall(br'"([^"]+)"\s+"([^"]*)"', block))
        source_class = raw_values.get(b"classname", b"").decode("ascii", errors="replace")
        if source_class.upper() != "PLACE_NAME":
            continue
        location_id = raw_values.get(b"id", b"").decode("ascii", errors="replace")
        raw_name = raw_values.get(b"name", b"")
        name = raw_name.decode("cp932", errors="replace")
        model = raw_values.get(b"model", b"").decode("ascii", errors="replace")
        row = {
            "id": location_id,
            "name": name,
            "model": model,
            "name_cp932_hex": raw_name.hex(),
        }
        entities.append(row)
        grouped = by_id.setdefault(location_id, {"id": location_id, "names": [], "models": []})
        if name not in grouped["names"]:
            grouped["names"].append(name)
        if model and model not in grouped["models"]:
            grouped["models"].append(model)

    def id_sort_key(item: dict) -> tuple[int, int | str]:
        value = item["id"]
        return (0, int(value)) if value.isdigit() else (1, value)

    payload = {
        "source_bsp": str(bsp_path),
        "bsp_version": version,
        "encoding": "CP932 (Shift-JIS)",
        "location_count": len(entities),
        "unique_id_count": len(by_id),
        "locations_by_id": sorted(by_id.values(), key=id_sort_key),
        "entities": entities,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"output": str(destination), "locations": len(entities), "unique_ids": len(by_id)}


def export_bsp_skybox(
    bsp_path: Path, hammer_dir: Path, sky_name: str, override: bool = False,
) -> dict:
    """Stage the six GoldSrc skybox faces and Source VMT definitions."""
    env_dir = bsp_path.parent.parent / "gfx" / "env"
    materialsrc = hammer_dir / "materialsrc" / "skybox"
    materials = hammer_dir / "materials" / "skybox"
    materialsrc.mkdir(parents=True, exist_ok=True)
    materials.mkdir(parents=True, exist_ok=True)
    files_by_stem = {
        item.stem.lower(): item for item in env_dir.iterdir()
        if item.is_file() and item.suffix.lower() in (".tga", ".bmp")
    } if env_dir.is_dir() else {}
    faces = []
    missing = []
    for suffix in ("bk", "dn", "ft", "lf", "rt", "up"):
        source = files_by_stem.get(f"{sky_name}{suffix}".lower())
        if source is None:
            missing.append(suffix)
            continue
        staged = materialsrc / f"{sky_name}{suffix}{source.suffix.lower()}"
        skipped = staged.is_file() and not override
        if not skipped:
            shutil.copyfile(source, staged)
        vmt = materials / f"{sky_name}{suffix}.vmt"
        vmt.write_text(
            '"UnlitGeneric"\n{\n'
            f'    "$basetexture" "skybox/{sky_name}{suffix}"\n'
            '    "$ignorez" "1"\n'
            '    "$nofog" "1"\n'
            '}\n',
            encoding="utf-8",
        )
        faces.append({
            "suffix": suffix, "source": str(source), "staged": str(staged),
            "material": str(vmt), "skipped_existing": skipped,
        })
    return {
        "sky_name": sky_name, "source_directory": str(env_dir),
        "faces": faces, "missing_faces": missing, "complete": not missing,
    }


def _source_point(point, center, scale):
    return (-(point[1] - center[1]) * scale,
            (point[0] - center[0]) * scale,
            (point[2] - center[2]) * scale)


def _source_angles(value: str) -> str:
    try:
        pitch, yaw, roll = map(float, value.split())
    except (ValueError, TypeError):
        return value
    yaw = (yaw + 90.0) % 360.0
    return f"{pitch:.9g} {yaw:.9g} {roll:.9g}"


def reconstruct_brush_faces(brush: list[dict]) -> bool:
    planes = [face["plane"] for face in brush]
    for first, second in itertools.combinations(planes, 2):
        alignment = _dot(first[0], second[0])
        same_distance = abs(first[1] - second[1]) if alignment > 0 else abs(first[1] + second[1])
        if abs(alignment) > 0.99999 and same_distance < 0.05:
            return False
    vertices: list[tuple[float, float, float]] = []
    for indices in itertools.combinations(range(len(planes)), 3):
        point = _intersection(*(planes[index] for index in indices))
        if (point is None or not all(math.isfinite(value) for value in point)
                or any(abs(value) > 32768 for value in point)):
            continue
        # GoldSrc MAP plane winding points toward the brush interior.
        if all(_dot(normal, point) >= distance - 0.05 for normal, distance in planes):
            if not any(sum((point[i] - old[i]) ** 2 for i in range(3)) < 0.01 for old in vertices):
                vertices.append(point)
    if len(vertices) < 4:
        return False
    for face in brush:
        normal, distance = face["plane"]
        polygon = [point for point in vertices if abs(_dot(normal, point) - distance) <= 0.1]
        if len(polygon) < 3:
            return False
        center = tuple(sum(point[i] for point in polygon) / len(polygon) for i in range(3))
        face["polygon"] = polygon
        face["center"] = center
    return True


def neo_surface_samples(neo: NeoFile, max_texture_size: int = 1024) -> list[dict]:
    vertices, texcoords, indices = neo.vertices(), neo.texcoords(), neo.indices()
    commands, texinfos = neo.draw_commands(), neo.texinfo()
    meshes = {record[2]: record for record in neo.meshes()}
    textures = {item.index: item for item in neo.texture_info()}
    samples: list[dict] = []
    for draw_id, command in enumerate(commands):
        mesh = meshes.get(draw_id)
        if mesh is None:
            continue
        info, texture_index, _, _ = resolve_mesh_material(neo, mesh, texinfos)
        diffuse_slots, _, _ = resolve_mesh_diffuse_slots(neo, mesh, texinfos)
        selector_start = diffuse_selector_start(mesh, info, texinfos, diffuse_slots)
        if info is None or texture_index is None:
            continue
        if command.indexed:
            base, runs = indexed_runs(command, indices)
            streams = [[
                (base + value, *mesh_texcoord_indices(info, selector_start, value))
                for value in run
            ] for run in runs]
        else:
            end = min(command.start + command.count, len(vertices))
            streams = [[
                (value, *mesh_texcoord_indices(info, selector_start, value - command.start))
                for value in range(command.start, end)
            ]]
        for stream in streams:
            for triangle in triangles_for_primitive(command.mode, stream):
                vi = [corner[0] for corner in triangle]
                ui = [corner[1] for corner in triangle]
                if len(set(vi)) != 3 or any(v < 0 or v >= len(vertices) for v in vi):
                    continue
                if any(u < 0 or u >= len(texcoords) for u in ui):
                    continue
                face_texture_index = selected_diffuse_texture(
                    diffuse_slots, texcoords, [corner[2] for corner in triangle]
                )
                if face_texture_index is None:
                    face_texture_index = texture_index
                texture = textures.get(face_texture_index)
                if texture is None:
                    continue
                points = [vertices[v] for v in vi]
                plane = _plane(points)
                if plane is None:
                    continue
                width, height = texture.width, texture.height
                if texture.external_reference:
                    source = neo.path.parent.parent / "tex" / Path(texture.external_reference)
                    if source.is_file():
                        header = source.read_bytes()[:128]
                        if header[:4] == b"DDS " and len(header) >= 20:
                            height, width = struct.unpack_from("<II", header, 12)
                        elif source.suffix.lower() == ".tga" and len(header) >= 18:
                            width, height = struct.unpack_from("<HH", header, 12)
                # Brush texture axes use VTF texel dimensions. Match the exact
                # power-of-two resizing performed before vtex compilation.
                width = 1 << (min(max(1, width), max_texture_size).bit_length() - 1)
                height = 1 << (min(max(1, height), max_texture_size).bit_length() - 1)
                samples.append({
                    "center": tuple(sum(point[i] for point in points) / 3 for i in range(3)),
                    "normal": plane[0], "distance": plane[1], "points": points,
                    "uv": [texcoords[u] for u in ui],
                    "material": f"neo_{texture.index:04d}_{texture.name}",
                    "width": max(1, width), "height": max(1, height),
                })
    return samples


def fitted_texture_axis(sample: dict, values: Sequence[float], center, scale: float) -> str | None:
    p0, p1, p2 = sample["points"]
    e1, e2 = _vsub(p1, p0), _vsub(p2, p0)
    aa, ab, bb = _dot(e1, e1), _dot(e1, e2), _dot(e2, e2)
    determinant = aa * bb - ab * ab
    if abs(determinant) < 1e-10:
        return None
    d1, d2 = values[1] - values[0], values[2] - values[0]
    first = (d1 * bb - d2 * ab) / determinant
    second = (d2 * aa - d1 * ab) / determinant
    gradient = tuple(first * e1[i] + second * e2[i] for i in range(3))
    magnitude = math.sqrt(_dot(gradient, gradient))
    if magnitude < 1e-10:
        return None
    # studiomdl/Source rotates native NEO XY as (-Y, X). VMF texture axes
    # must use the same coordinate system as the transformed brush planes.
    source_gradient = (-gradient[1], gradient[0], gradient[2])
    axis = tuple(value / magnitude for value in source_gradient)
    axis_scale = scale / magnitude
    offset = values[0] - _dot(gradient, p0) + _dot(gradient, center)
    return f"[{axis[0]:.9g} {axis[1]:.9g} {axis[2]:.9g} {offset:.9g}] {axis_scale:.9g}"


def _projection_axis(normal: tuple[float, float, float]) -> int:
    return max(range(3), key=lambda axis: abs(normal[axis]))


def _project_2d(point: tuple[float, float, float], dropped: int) -> tuple[float, float]:
    axes = [axis for axis in range(3) if axis != dropped]
    return point[axes[0]], point[axes[1]]


def _polygon_area_2d(polygon: Sequence[tuple[float, float]]) -> float:
    return abs(sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
        for index in range(len(polygon))
    )) * 0.5


def _convex_hull_2d(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) <= 2:
        return unique
    def turn(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], point) <= 1e-8:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], point) <= 1e-8:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _convex_intersection_area(subject, clip) -> float:
    """Sutherland-Hodgman intersection area for counter-clockwise polygons."""
    output = list(subject)
    for index, edge_start in enumerate(clip):
        edge_end = clip[(index + 1) % len(clip)]
        input_polygon, output = output, []
        if not input_polygon:
            break
        def side(point):
            return ((edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
                    - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0]))
        previous = input_polygon[-1]
        previous_inside = side(previous) >= -1e-7
        for current in input_polygon:
            current_inside = side(current) >= -1e-7
            if current_inside != previous_inside:
                dx, dy = current[0] - previous[0], current[1] - previous[1]
                ex, ey = edge_end[0] - edge_start[0], edge_end[1] - edge_start[1]
                denominator = dx * ey - dy * ex
                if abs(denominator) > 1e-12:
                    t = ((edge_start[0] - previous[0]) * ey
                         - (edge_start[1] - previous[1]) * ex) / denominator
                    output.append((previous[0] + t * dx, previous[1] + t * dy))
            if current_inside:
                output.append(current)
            previous, previous_inside = current, current_inside
    return _polygon_area_2d(output) if len(output) >= 3 else 0.0


def export_collision_hybrid_vmf(
    neo: NeoFile, map_path: Path, model_vmf: Path, destination: Path, scale: float
) -> dict:
    """Add invisible decompiled world collision to the textured model VMF."""
    brushes = parse_decompiled_world_brushes(map_path)
    vertices = neo.vertices()
    logical = neo.lumps[1].count or len(vertices)
    position_vertices = vertices[:min(logical, len(vertices))]
    center = tuple(
        (min(point[i] for point in position_vertices) + max(point[i] for point in position_vertices)) * 0.5
        for i in range(3)
    )
    solids: list[str] = []
    invalid = 0
    solid_id, side_id = 1_000_000, 2_000_000
    for brush in brushes:
        if not reconstruct_brush_faces(brush):
            invalid += 1
            continue
        sides: list[str] = []
        for face in brush:
            polygon = face["polygon"]
            best = None
            best_area = 0.0
            for candidate in itertools.combinations(polygon, 3):
                cross = _cross(_vsub(candidate[1], candidate[0]), _vsub(candidate[2], candidate[0]))
                area = math.sqrt(_dot(cross, cross))
                if area > best_area:
                    best, best_area = candidate, area
            if best is None or best_area < 1e-5:
                sides = []
                break
            p0, p1, p2 = best
            if _dot(_cross(_vsub(p1, p0), _vsub(p2, p0)), face["plane"][0]) < 0:
                p1, p2 = p2, p1
            transformed = [
                (
                    -(point[1] - center[1]) * scale,
                    (point[0] - center[0]) * scale,
                    (point[2] - center[2]) * scale,
                )
                for point in (p0, p1, p2)
            ]
            plane_text = " ".join("(" + " ".join(f"{value:.9g}" for value in point) + ")" for point in transformed)
            sides.append(
                f'side\n{{\n"id" "{side_id}"\n"plane" "{plane_text}"\n'
                '"material" "TOOLS/TOOLSNODRAW"\n'
                '"uaxis" "[1 0 0 0] 0.25"\n"vaxis" "[0 -1 0 0] 0.25"\n'
                '"rotation" "0"\n"lightmapscale" "16"\n"smoothing_groups" "0"\n}\n'
            )
            side_id += 1
        if sides:
            solids.append(f'solid\n{{\n"id" "{solid_id}"\n' + "".join(sides) + '}\n')
            solid_id += 1
        else:
            invalid += 1

    vmf_text = model_vmf.read_text(encoding="utf-8")
    world_start = vmf_text.find("world\n{")
    if world_start < 0:
        raise NeoFormatError(f"could not find world block in {model_vmf}")
    depth = 0
    world_end = None
    for pos in range(vmf_text.find("{", world_start), len(vmf_text)):
        if vmf_text[pos] == "{":
            depth += 1
        elif vmf_text[pos] == "}":
            depth -= 1
            if depth == 0:
                world_end = pos
                break
    if world_end is None:
        raise NeoFormatError(f"unterminated world block in {model_vmf}")
    destination.write_text(
        vmf_text[:world_end] + "".join(solids) + vmf_text[world_end:], encoding="utf-8"
    )
    return {
        "output": str(destination), "input_brushes": len(brushes),
        "collision_brushes": len(solids), "invalid_brushes": invalid,
        "material": "TOOLS/TOOLSNODRAW",
    }


def export_decompiled_vmf(
    neo: NeoFile, map_path: Path, destination: Path, scale: float,
    max_texture_size: int = 1024, target_game: str = "gmod",
    transfer_lights: bool = False, sky_name: str = "sky_day01_01",
) -> dict:
    if map_path.suffix.lower() == ".vmf":
        map_entities = parse_decompiled_vmf_entities(map_path)
        brushes = map_entities[0]["brushes"]
    else:
        brushes = parse_decompiled_world_brushes(map_path)
        map_entities = parse_decompiled_entities(map_path)
    samples = neo_surface_samples(neo, max_texture_size)
    grid: dict[tuple[int, int, int], list[dict]] = {}
    grid_size = 256.0
    for sample in samples:
        low = [math.floor(min(point[axis] for point in sample["points"]) / grid_size) for axis in range(3)]
        high = [math.floor(max(point[axis] for point in sample["points"]) / grid_size) for axis in range(3)]
        cells = math.prod(high[axis] - low[axis] + 1 for axis in range(3))
        if cells <= 64:
            for key in itertools.product(*(range(low[axis], high[axis] + 1) for axis in range(3))):
                grid.setdefault(key, []).append(sample)
        else:
            key = tuple(math.floor(value / grid_size) for value in sample["center"])
            grid.setdefault(key, []).append(sample)

    vertices = neo.vertices()
    logical = neo.lumps[1].count or len(vertices)
    position_vertices = vertices[:min(logical, len(vertices))]
    center = tuple(
        (min(point[i] for point in position_vertices) + max(point[i] for point in position_vertices)) * 0.5
        for i in range(3)
    )
    special = {
        "null": "TOOLS/TOOLSNODRAW", "clip": "TOOLS/TOOLSCLIP",
        "aaatrigger": "TOOLS/TOOLSTRIGGER", "sky": "TOOLS/TOOLSSKYBOX",
    }
    matched = unmatched = invalid = 0
    confidence_rows: list[dict] = []
    brush_materials: set[str] = set()
    solids: list[str] = []
    solid_id, side_id = 2, 100000
    for brush in brushes:
        if not reconstruct_brush_faces(brush):
            invalid += 1
            continue
        sides: list[str] = []
        for face in brush:
            original = face["texture"].lower()
            material = special.get(original)
            match = None
            if material is None and original.startswith("!"):
                material = "TOOLS/TOOLSNODRAW"
            if material is None:
                c = face["center"]
                polygon = face["polygon"]
                low = [math.floor((min(point[axis] for point in polygon) - 16) / grid_size) for axis in range(3)]
                high = [math.floor((max(point[axis] for point in polygon) + 16) / grid_size) for axis in range(3)]
                candidates_by_id: dict[int, dict] = {}
                cells = math.prod(high[axis] - low[axis] + 1 for axis in range(3))
                if cells <= 512:
                    for key in itertools.product(*(range(low[axis], high[axis] + 1) for axis in range(3))):
                        for candidate in grid.get(key, ()):
                            candidates_by_id[id(candidate)] = candidate
                else:
                    cell = tuple(math.floor(value / grid_size) for value in c)
                    for delta in itertools.product(range(-2, 3), repeat=3):
                        for candidate in grid.get(tuple(cell[i] + delta[i] for i in range(3)), ()):
                            candidates_by_id[id(candidate)] = candidate
                fn, _ = face["plane"]
                dropped = _projection_axis(fn)
                face_2d = _convex_hull_2d([_project_2d(point, dropped) for point in polygon])
                face_area = _polygon_area_2d(face_2d)
                material_areas: dict[str, float] = {}
                material_samples: dict[str, list[tuple[float, dict]]] = {}
                for sample in candidates_by_id.values():
                    alignment = abs(_dot(fn, sample["normal"]))
                    plane_error = abs(abs(_dot(sample["normal"], c) - sample["distance"]))
                    if alignment < 0.985 or plane_error > 4.0:
                        continue
                    triangle_2d = _convex_hull_2d([
                        _project_2d(point, dropped) for point in sample["points"]
                    ])
                    overlap = _convex_intersection_area(triangle_2d, face_2d)
                    if overlap <= 1e-5:
                        continue
                    weighted = overlap * alignment / (1.0 + plane_error)
                    name = sample["material"]
                    material_areas[name] = material_areas.get(name, 0.0) + overlap
                    material_samples.setdefault(name, []).append((weighted, sample))
                total_overlap = sum(material_areas.values())
                winner = max(material_areas, key=material_areas.get) if material_areas else None
                winner_area = material_areas.get(winner, 0.0) if winner else 0.0
                coverage = min(1.0, total_overlap / face_area) if face_area > 1e-6 else 0.0
                dominance = winner_area / total_overlap if total_overlap > 1e-6 else 0.0
                confidence = coverage * dominance
                if winner and coverage >= 0.30 and dominance >= 0.60 and confidence >= 0.24:
                    match = max(material_samples[winner], key=lambda item: item[0])[1]
                    base_material = f"neo_maps/{neo.path.stem}/{match['material']}"
                    material = base_material + "_brush"
                    brush_materials.add(match["material"])
                    matched += 1
                else:
                    material = "NEO_TRANSFER/MANUAL_MISSING"
                    unmatched += 1
                confidence_rows.append({
                    "original_texture": face["texture"], "center": list(c),
                    "material": match["material"] if match else None,
                    "coverage": round(coverage, 5), "dominance": round(dominance, 5),
                    "confidence": round(confidence, 5), "accepted": match is not None,
                })

            polygon = face["polygon"]
            best = None
            best_area = 0.0
            for candidate in itertools.combinations(polygon, 3):
                area = math.sqrt(_dot(_cross(_vsub(candidate[1], candidate[0]), _vsub(candidate[2], candidate[0])),
                                      _cross(_vsub(candidate[1], candidate[0]), _vsub(candidate[2], candidate[0]))))
                if area > best_area:
                    best, best_area = candidate, area
            if best is None or best_area < 1e-5:
                sides = []
                break
            p0, p1, p2 = best
            fn = face["plane"][0]
            if _dot(_cross(_vsub(p1, p0), _vsub(p2, p0)), fn) < 0:
                p1, p2 = p2, p1
            transformed = [
                (
                    -(p[1] - center[1]) * scale,
                    (p[0] - center[0]) * scale,
                    (p[2] - center[2]) * scale,
                )
                for p in (p0, p1, p2)
            ]
            plane_text = " ".join("(" + " ".join(f"{v:.9g}" for v in p) + ")" for p in transformed)
            uaxis, vaxis = "[1 0 0 0] 0.25", "[0 -1 0 0] 0.25"
            if match is not None:
                u_values = [uv[0] * match["width"] for uv in match["uv"]]
                # VMF texture axes use Source's world-projection convention;
                # unlike SMD/OBJ UVs, applying 1-V here vertically mirrors the
                # fitted brush material. Preserve the native NEO V direction.
                v_values = [uv[1] * match["height"] for uv in match["uv"]]
                uaxis = fitted_texture_axis(match, u_values, center, scale) or uaxis
                vaxis = fitted_texture_axis(match, v_values, center, scale) or vaxis
            sides.append(
                f'side\n{{\n"id" "{side_id}"\n"plane" "{plane_text}"\n'
                f'"material" "{material}"\n"uaxis" "{uaxis}"\n"vaxis" "{vaxis}"\n'
                '"rotation" "0"\n"lightmapscale" "16"\n"smoothing_groups" "0"\n}\n'
            )
            side_id += 1
        if sides:
            solids.append(f'solid\n{{\n"id" "{solid_id}"\n' + "".join(sides) + '}\n')
            solid_id += 1
        else:
            invalid += 1

    entity_class_map = {
        "info_player_start": "info_player_start",
        "info_player_deathmatch": "info_player_deathmatch",
        "light_environment": "light_environment",
        "env_sprite": "env_sprite",
        "trigger_push": "trigger_push",
        "trigger_relay": "logic_relay",
        "training_trigger_once": "trigger_once",
        "training_breakable": "func_breakable",
        "training_door": "func_door",
        "func_ladder": "func_ladder",
        "func_illusionary": "func_brush",
        "func_wall": "func_brush",
        "func_water": "func_water_analog",
        "func_buyzone": "func_buyzone",
        "path_corner": "path_corner",
        "trigger_changetarget": "trigger_changetarget",
        "multi_manager": "logic_relay",
        "trigger_multiple": "trigger_multiple",
        "func_wall_toggle": "func_brush",
        "func_breakable": "func_breakable",
        "light": "light",
        "func_button": "func_button",
        "info_teleport_destination": "info_teleport_destination",
        "func_door": "func_door",
        "ambient_generic": "ambient_generic",
        "trigger_teleport": "trigger_teleport",
        "func_bomb_target": "func_bomb_target",
        "env_explosion": "env_explosion",
        "trigger_once": "trigger_once",
        "game_zone_player": "game_zone_player",
        "game_counter": "math_counter",
        "trigger_hurt": "trigger_hurt",
        "func_conveyor": "func_conveyor",
        "func_rotating": "func_rotating",
        "game_counter_set": "math_counter",
        "env_render": "env_render",
        "env_shake": "env_shake",
        "button_target": "func_button",
        "env_spark": "env_spark",
        "info_target": "info_target",
        "trigger_camera": "point_viewcontrol",
        "training_wall_toggle": "func_brush",
        # NEO uses brush volumes to associate areas of a map with location
        # names shown by its HUD.  Source has no directly portable generic
        # PLACE_NAME entity, so retain the volume as a non-solid trigger.  The
        # original name/id are copied to metadata below for Hammer and scripts.
        "PLACE_NAME": "trigger_multiple",
    }
    if target_game == "css":
        # GoldSrc Counter-Strike uses start for CT and deathmatch for T.
        # Counter-Strike: Source requires explicit team spawn classnames.
        entity_class_map.update({
            "info_player_start": "info_player_counterterrorist",
            "info_player_deathmatch": "info_player_terrorist",
        })
    brush_material_by_class = {
        # Garry's Mod does not ship the SDK's TOOLSLADDER VMT. The func_ladder
        # entity supplies ladder behavior; keep its volume invisible with nodraw.
        "func_ladder": "TOOLS/TOOLSNODRAW",
        "trigger_push": "TOOLS/TOOLSTRIGGER",
        "trigger_once": "TOOLS/TOOLSTRIGGER",
        "trigger_multiple": "TOOLS/TOOLSTRIGGER",
        "trigger_hurt": "TOOLS/TOOLSTRIGGER",
        "trigger_teleport": "TOOLS/TOOLSTRIGGER",
        "PLACE_NAME": "TOOLS/TOOLSTRIGGER",
        "game_zone_player": "TOOLS/TOOLSTRIGGER",
        "func_buyzone": "TOOLS/TOOLSTRIGGER",
        "func_bomb_target": "TOOLS/TOOLSTRIGGER",
        "func_water_analog": "TOOLS/TOOLSNODRAW",
    }
    ignored_keys = {"classname", "id", "wad", "mapversion", "rendermode", "renderamt", "rendercolor"}
    targets_by_name: dict[str, str] = {}
    for item in map_entities[1:]:
        item_values = item["keyvalues"]
        name = item_values.get("targetname")
        mapped = entity_class_map.get(item_values.get("classname", ""))
        if name and mapped:
            targets_by_name[name] = mapped

    def target_input(name: str) -> str:
        return {
            "logic_relay": "Trigger", "func_door": "Toggle", "func_brush": "Toggle",
            "func_button": "Press", "ambient_generic": "ToggleSound",
            "env_sprite": "ToggleSprite", "func_breakable": "Break",
            "math_counter": "Add", "env_explosion": "Explode",
        }.get(targets_by_name.get(name, ""), "Trigger")

    entity_texts: list[str] = []
    unsupported_classes: dict[str, int] = {}
    converted_classes: dict[str, int] = {}
    manual_review_rows: list[dict] = []
    entity_id = max(solid_id + 1, 500000)
    entity_side_id = max(side_id + 1, 600000)
    source_water_material = "NATURE/WATER_CANALS_CITY"
    converted_water_brushes = 0
    ambiguous_water_brushes = 0
    for source_entity in map_entities[1:]:
        values = source_entity["keyvalues"]
        source_class = values.get("classname", "")
        target_class = entity_class_map.get(source_class)
        if target_class is None:
            unsupported_classes[source_class or "<missing>"] = unsupported_classes.get(source_class or "<missing>", 0) + 1
            manual_review_rows.append({"classname": source_class or "<missing>",
                                       "keyvalues": values,
                                       "brushes": len(source_entity["brushes"])})
            # Preserve uncertain brush entities as visible manual-review
            # func_brushes. Preserve point entities only when they have a useful
            # origin; metadata-only helpers remain in the JSON review report.
            if source_entity["brushes"]:
                target_class = "func_brush"
            elif "origin" in values:
                target_class = "info_target"
            else:
                continue
        converted_classes[target_class] = converted_classes.get(target_class, 0) + 1
        parts = [f'entity\n{{\n"id" "{entity_id}"\n"classname" "{target_class}"\n']
        if source_class == "PLACE_NAME":
            place_id = values.get("id", "unknown")
            original_name = values.get("name", "")
            safe_place_id = re.sub(r"[^A-Za-z0-9_]+", "_", place_id).strip("_") or "unknown"
            parts.append(f'"targetname" "neo_location_{safe_place_id}_{entity_id}"\n')
            parts.append(f'"_neo_location_id" "{place_id.replace(chr(34), chr(39))}"\n')
            if original_name:
                parts.append(f'"_neo_location_name" "{original_name.replace(chr(34), chr(39))}"\n')
            # A negative wait prevents repeated automatic firing.  Outputs can
            # be added manually later if a port needs location notifications.
            parts.append('"spawnflags" "1"\n')
            parts.append('"wait" "-1"\n')
        output_target = values.get("target")
        for key, value in values.items():
            if key.lower() in ignored_keys:
                continue
            if key in {"target", "killtarget"}:
                continue
            if source_class == "PLACE_NAME" and key in {"name", "id", "targetname", "wait"}:
                continue
            if key == "origin":
                try:
                    point = tuple(map(float, value.split()))
                    value = " ".join(f"{coordinate:.9g}" for coordinate in _source_point(point, center, scale))
                except ValueError:
                    pass
            elif key == "angles":
                value = _source_angles(value)
            # Common GoldSrc-to-Source key translations.
            if target_game == "css" and source_class == "func_buyzone" and key.lower() == "team":
                # GoldSrc CS uses 1=T and 2=CT. CS:S's TeamNum base uses
                # 2=Terrorist and 3=Counter-Terrorist.
                target_key = "TeamNum"
                value = {"1": "2", "2": "3"}.get(value.strip(), value)
            elif key == "spawnflags":
                target_key = "_neo_spawnflags_original"
            elif key == "master":
                target_key = "_neo_master_manual"
            else:
                target_key = {"movesnd": "movesnd", "health": "health", "speed": "speed",
                              "wait": "wait", "targetname": "targetname"}.get(key, key)
            escaped = value.replace('"', "'")
            parts.append(f'"{target_key}" "{escaped}"\n')
        manual_review_entity = source_class not in entity_class_map
        if manual_review_entity:
            parts.append(f'"_neo_original_class" "{source_class}"\n"_neo_manual_review" "1"\n')
        if output_target:
            output_name = {
                "logic_relay": "OnTrigger", "trigger_once": "OnTrigger",
                "trigger_multiple": "OnTrigger", "trigger_push": "OnStartTouch",
                "func_button": "OnPressed", "func_breakable": "OnBreak",
                "func_door": "OnOpen", "math_counter": "OnHitMax",
            }.get(target_class, "OnTrigger")
            parts.append(f'"{output_name}" "{output_target},{target_input(output_target)},,0,-1"\n')
        if source_class == "multi_manager":
            standard = {"classname", "origin", "targetname", "spawnflags", "master"}
            for target, delay in values.items():
                if target in standard:
                    continue
                try:
                    float(delay)
                except ValueError:
                    continue
                parts.append(f'"OnTrigger" "{target},{target_input(target)},,{delay},-1"\n')
        if target_class == "func_brush" and source_class == "func_illusionary":
            parts.append('"Solidity" "1"\n')
        elif target_class == "func_brush" and manual_review_entity:
            # Unknown brush triggers/helpers are retained for inspection but
            # must not create accidental invisible collision in the port.
            parts.append('"Solidity" "1"\n')
        if target_class == "func_breakable" and "material" not in values:
            parts.append('"material" "0"\n')
        entity_brushes = 0
        for brush_index, brush in enumerate(source_entity["brushes"]):
            if not reconstruct_brush_faces(brush):
                continue
            water_top_index = None
            if target_class == "func_water_analog":
                horizontal_faces = [
                    (index, sum(point[2] for point in face["polygon"]) / len(face["polygon"]))
                    for index, face in enumerate(brush)
                    if face.get("polygon") and abs(face["plane"][0][2]) >= 0.7
                ]
                if horizontal_faces:
                    water_top_index = max(horizontal_faces, key=lambda item: item[1])[0]
                    converted_water_brushes += 1
                else:
                    ambiguous_water_brushes += 1
                    manual_review_rows.append({
                        "classname": source_class,
                        "reason": "water brush has no near-horizontal top face",
                        "entity_id": entity_id,
                        "brush_index": brush_index,
                    })
            brush_sides: list[str] = []
            for face_index, face in enumerate(brush):
                polygon = face["polygon"]
                best = max(
                    itertools.combinations(polygon, 3),
                    key=lambda candidate: _dot(
                        _cross(_vsub(candidate[1], candidate[0]), _vsub(candidate[2], candidate[0])),
                        _cross(_vsub(candidate[1], candidate[0]), _vsub(candidate[2], candidate[0])),
                    ),
                    default=None,
                )
                if best is None:
                    brush_sides = []
                    break
                p0, p1, p2 = best
                if _dot(_cross(_vsub(p1, p0), _vsub(p2, p0)), face["plane"][0]) < 0:
                    p1, p2 = p2, p1
                transformed = [_source_point(point, center, scale) for point in (p0, p1, p2)]
                plane_text = " ".join("(" + " ".join(f"{v:.9g}" for v in p) + ")" for p in transformed)
                if target_class == "func_water_analog" and face_index == water_top_index:
                    face_material = source_water_material
                else:
                    face_material = brush_material_by_class.get(
                        target_class, "NEO_TRANSFER/MANUAL_MISSING"
                    )
                brush_sides.append(
                    f'side\n{{\n"id" "{entity_side_id}"\n"plane" "{plane_text}"\n'
                    f'"material" "{face_material}"\n'
                    '"uaxis" "[1 0 0 0] 0.25"\n"vaxis" "[0 -1 0 0] 0.25"\n'
                    '"rotation" "0"\n"lightmapscale" "16"\n"smoothing_groups" "0"\n}\n'
                )
                entity_side_id += 1
            if brush_sides:
                parts.append(f'solid\n{{\n"id" "{entity_id + entity_brushes + 1}"\n' + "".join(brush_sides) + '}\n')
                entity_brushes += 1
        parts.append('}\n')
        entity_texts.append("".join(parts))
        entity_id += max(1, entity_brushes + 1)

    generated_buyzones = 0
    source_buyzones = sum(
        entity["keyvalues"].get("classname") == "func_buyzone"
        for entity in map_entities[1:]
    )
    if target_game == "css" and source_buyzones == 0:
        spawn_groups = (
            ("info_player_deathmatch", 2),  # Terrorist
            ("info_player_start", 3),       # Counter-Terrorist
        )
        for spawn_class, team_num in spawn_groups:
            spawn_points: list[tuple[float, float, float]] = []
            for source_entity in map_entities[1:]:
                values = source_entity["keyvalues"]
                if values.get("classname") != spawn_class or "origin" not in values:
                    continue
                try:
                    raw_point = tuple(map(float, values["origin"].split()))
                    if len(raw_point) == 3:
                        spawn_points.append(_source_point(raw_point, center, scale))
                except ValueError:
                    continue
            if not spawn_points:
                continue
            parts = [
                f'entity\n{{\n"id" "{entity_id}"\n"classname" "func_buyzone"\n'
                f'"TeamNum" "{team_num}"\n'
            ]
            for point in spawn_points:
                x0, x1 = point[0] - 128, point[0] + 128
                y0, y1 = point[1] - 128, point[1] + 128
                z0, z1 = point[2] - 32, point[2] + 96
                planes = (
                    ((x1, y1, z1), (x1, y1, z0), (x1, y0, z1)),
                    ((x0, y0, z1), (x0, y0, z0), (x0, y1, z1)),
                    ((x0, y1, z1), (x0, y1, z0), (x1, y1, z1)),
                    ((x1, y0, z1), (x1, y0, z0), (x0, y0, z1)),
                    ((x0, y0, z1), (x0, y1, z1), (x1, y0, z1)),
                    ((x0, y1, z0), (x0, y0, z0), (x1, y1, z0)),
                )
                sides: list[str] = []
                for plane in planes:
                    plane_text = " ".join(
                        "(" + " ".join(f"{value:.9g}" for value in vertex) + ")"
                        for vertex in plane
                    )
                    sides.append(
                        f'side\n{{\n"id" "{entity_side_id}"\n"plane" "{plane_text}"\n'
                        '"material" "TOOLS/TOOLSTRIGGER"\n'
                        '"uaxis" "[1 0 0 0] 0.25"\n"vaxis" "[0 -1 0 0] 0.25"\n'
                        '"rotation" "0"\n"lightmapscale" "16"\n"smoothing_groups" "0"\n}\n'
                    )
                    entity_side_id += 1
                parts.append(
                    f'solid\n{{\n"id" "{entity_id + len(parts)}"\n' + "".join(sides) + '}\n'
                )
            parts.append('}\n')
            entity_texts.append("".join(parts))
            entity_id += len(spawn_points) + 1
            converted_classes["func_buyzone"] = converted_classes.get("func_buyzone", 0) + 1
            generated_buyzones += 1

    light_review_rows: list[dict] = []
    transferred_lights = 0
    if transfer_lights:
        for light in neo.lights():
            values = (*light.position, light.intensity, *light.diffuse)
            if not all(math.isfinite(value) for value in values):
                light_review_rows.append({
                    "index": light.index, "status": "skipped_non_finite",
                    "position": list(light.position), "intensity": light.intensity,
                })
                continue
            # Negative records represent subtractive renderer lights. VRAD's
            # ordinary Source light entity has no reliable equivalent; adding
            # their absolute value would brighten exactly the areas they were
            # intended to darken.
            if light.intensity <= 0:
                light_review_rows.append({
                    "index": light.index, "status": "manual_subtractive_or_disabled",
                    "position": list(light.position), "intensity": light.intensity,
                    "diffuse": list(light.diffuse),
                })
                continue
            rgb = tuple(max(0, min(255, round(channel * 255))) for channel in light.diffuse[:3])
            if not any(rgb):
                light_review_rows.append({
                    "index": light.index, "status": "skipped_black",
                    "position": list(light.position), "intensity": light.intensity,
                })
                continue
            # Most maps use values around 100-400. A handful contain values as
            # high as 70000; preserving those literally makes VRAD output and
            # the resulting map unusable, so retain the raw value as metadata.
            brightness = min(2000.0, light.intensity)
            origin = _source_point(light.position, center, scale)
            entity_texts.append(
                f'entity\n{{\n"id" "{entity_id}"\n"classname" "light"\n'
                f'"origin" "{" ".join(f"{value:.9g}" for value in origin)}"\n'
                f'"_light" "{rgb[0]} {rgb[1]} {rgb[2]} {brightness:.9g}"\n'
                '}\n'
            )
            light_review_rows.append({
                "index": light.index,
                "status": "brightness_clamped" if brightness != light.intensity else "transferred",
                "position": list(light.position), "intensity": light.intensity,
                "exported_origin": list(origin), "exported_rgb": list(rgb),
                "exported_brightness": brightness,
                "secondary": list(light.secondary), "ambient": list(light.ambient),
                "reserved": list(light.reserved),
            })
            entity_id += 1
            transferred_lights += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    material_dir = destination.parent / "materials" / "neo_maps" / neo.path.stem
    material_dir.mkdir(parents=True, exist_ok=True)
    for material_name in brush_materials:
        (material_dir / f"{material_name}_brush.vmt").write_text(
            '"LightmappedGeneric"\n{\n'
            f'    "$basetexture" "models/neo_maps/{neo.path.stem}/{material_name}"\n'
            '}\n',
            encoding="utf-8",
        )
    destination.write_text(
        'versioninfo\n{\n"editorversion" "400"\n"editorbuild" "0"\n"mapversion" "1"\n'
        '"formatversion" "100"\n"prefab" "0"\n}\n'
        'visgroups\n{\n}\nviewsettings\n{\n"bSnapToGrid" "1"\n"bShowGrid" "1"\n'
        '"bShowLogicalGrid" "0"\n"nGridSpacing" "16"\n"bShow3DGrid" "0"\n}\n'
        'world\n{\n"id" "1"\n"mapversion" "1"\n"classname" "worldspawn"\n'
        f'"skyname" "{sky_name}"\n' + "".join(solids) + '}\n'
        + "".join(entity_texts) +
        'cameras\n{\n"activecamera" "-1"\n}\ncordons\n{\n"active" "0"\n}\n',
        encoding="utf-8",
    )
    confidence_path = destination.with_name(destination.stem + "_matches.json")
    confidence_path.write_text(json.dumps(confidence_rows, indent=2), encoding="utf-8")
    entity_review_path = destination.with_name(destination.stem + "_entity_review.json")
    entity_review_path.write_text(json.dumps(manual_review_rows, indent=2), encoding="utf-8")
    light_review_path = destination.with_name(destination.stem + "_light_review.json")
    if transfer_lights:
        light_review_path.write_text(json.dumps(light_review_rows, indent=2), encoding="utf-8")
    return {"output": str(destination), "input_brushes": len(brushes), "written_brushes": len(solids),
            "invalid_brushes": invalid, "matched_faces": matched, "unmatched_faces": unmatched,
            "neo_surface_samples": len(samples), "brush_materials": len(brush_materials),
            "match_report": str(confidence_path), "entities": len(entity_texts),
            "converted_entity_classes": converted_classes,
            "unsupported_entity_classes": unsupported_classes,
            "entity_review": str(entity_review_path), "target_game": target_game,
            "transferred_lights": transferred_lights,
            "generated_buyzones": generated_buyzones,
            "water_material": source_water_material,
            "converted_water_brushes": converted_water_brushes,
            "ambiguous_water_brushes": ambiguous_water_brushes,
            "light_review": str(light_review_path) if transfer_lights else None}


def find_game_directory(source_bin: Path, override: Path | None) -> Path:
    if override is not None:
        game_dir = override.resolve()
        if not (game_dir / "gameinfo.txt").is_file():
            raise NeoFormatError(f"game directory has no gameinfo.txt: {game_dir}")
        return game_dir
    root = source_bin.resolve().parent
    candidates = [root, root / root.name.lower(), root / "garrysmod"]
    candidates.extend(path for path in root.iterdir() if path.is_dir())
    for candidate in candidates:
        if (candidate / "gameinfo.txt").is_file():
            return candidate
    raise NeoFormatError(
        f"could not infer the game directory below {root}; pass --game-dir explicitly"
    )


def export_css_overview(
    neo: NeoFile, overview_path: Path, hammer_dir: Path, scale: float,
    map_name: str | None = None,
    radar_rotation: str = "none",
) -> dict:
    """Convert a GoldSrc overview definition into staged CS:S overview assets."""
    text = overview_path.read_text(encoding="latin-1")

    def scalar(name: str) -> float:
        match = re.search(rf"\b{name}\s+([-+]?\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if not match:
            raise NeoFormatError(f"{overview_path}: missing {name} overview value")
        return float(match.group(1))

    origin_match = re.search(
        r"\bORIGIN\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)",
        text, re.IGNORECASE,
    )
    image_match = re.search(r'\bIMAGE\s+"([^"]+)"', text, re.IGNORECASE)
    if not origin_match or not image_match:
        raise NeoFormatError(f"{overview_path}: missing ORIGIN or IMAGE overview value")
    zoom = scalar("ZOOM")
    if zoom <= 0:
        raise NeoFormatError(f"{overview_path}: ZOOM must be greater than zero")
    original_origin = tuple(float(value) for value in origin_match.groups())
    original_rotated = int(scalar("ROTATED")) != 0
    image_reference = image_match.group(1).replace("\\", "/")
    image_name = Path(image_reference).name
    candidates = [overview_path.parent / image_name]
    if overview_path.parent.name.lower() != "overviews":
        candidates.append(overview_path.parent.parent / "overviews" / image_name)
    image_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if image_path is None:
        raise NeoFormatError(
            f"{overview_path}: referenced overview image was not found: {image_reference}"
        )

    vertices = neo.vertices()
    logical = neo.lumps[1].count or len(vertices)
    positions = vertices[:min(logical, len(vertices))]
    center = tuple(
        (min(point[axis] for point in positions) + max(point[axis] for point in positions)) * 0.5
        for axis in range(3)
    )
    source_center = _source_point(original_origin, center, scale)
    # Convert GoldSrc's overview zoom to Source world units per radar pixel.
    # The captured 1024-wide overview uses an 8192-unit horizontal span.
    source_scale = 8.0 * scale / zoom
    pos_x = source_center[0] - 512.0 * source_scale
    pos_y = source_center[1] + 512.0 * source_scale
    # Decompiled brush conversion rotates the original XY plane by +90 degrees.
    # Toggle GoldSrc's rotation flag to retain the captured image orientation.
    source_rotated = not original_rotated
    map_name = map_name or neo.path.stem

    materialsrc = hammer_dir / "materialsrc" / "overviews"
    materials = hammer_dir / "materials" / "overviews"
    resources = hammer_dir / "resource" / "overviews"
    for directory in (materialsrc, materials, resources):
        directory.mkdir(parents=True, exist_ok=True)
    staged_image = materialsrc / f"{map_name}{image_path.suffix.lower()}"
    shutil.copyfile(image_path, staged_image)
    (materials / f"{map_name}.vmt").write_text(
        '"UnlitGeneric"\n{\n'
        f'    "$basetexture" "overviews/{map_name}"\n'
        '    "$translucent" "1"\n    "$vertexalpha" "1"\n'
        '    "$vertexcolor" "1"\n    "$ignorez" "1"\n}\n',
        encoding="utf-8",
    )
    definition = resources / f"{map_name}.txt"
    definition.write_text(
        f'"{map_name}"\n{{\n'
        f'    "material" "overviews/{map_name}"\n'
        f'    "pos_x" "{pos_x:.9g}"\n'
        f'    "pos_y" "{pos_y:.9g}"\n'
        f'    "scale" "{source_scale:.9g}"\n'
        f'    "rotate" "{int(source_rotated)}"\n'
        '    "zoom" "1.0"\n}\n',
        encoding="utf-8",
    )
    return {
        "input": str(overview_path), "image": str(image_path),
        "staged_image": str(staged_image), "definition": str(definition),
        "material": str(materials / f"{map_name}.vmt"),
        "goldsrc_zoom": zoom, "goldsrc_origin": list(original_origin),
        "goldsrc_rotated": original_rotated, "pos_x": pos_x, "pos_y": pos_y,
        "scale": source_scale, "rotate": source_rotated,
        "image_rotation": radar_rotation,
    }


def run_source_tool(command: list[str], cwd: Path, timeout: float) -> dict:
    try:
        process = subprocess.run(
            command, cwd=cwd, text=True, capture_output=True, check=False,
            stdin=subprocess.DEVNULL, timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return {
            "command": command,
            "exit_code": -1,
            "timed_out": True,
            "timeout_seconds": timeout,
            "stdout": stdout[-8000:],
            "stderr": stderr[-8000:],
        }
    return {
        "command": command,
        "exit_code": process.returncode,
        "stdout": process.stdout[-8000:],
        "stderr": process.stderr[-8000:],
    }


def compile_hammer_output(
    hammer_dir: Path,
    source_bin: Path,
    game_override: Path | None,
    ffmpeg_override: Path | None,
    tool_timeout: float,
    verbose: bool,
    max_vtf_size: int,
    compile_models: bool = True,
    override_textures: bool = False,
    override_models: bool = False,
    radar_rotation: str = "none",
) -> dict:
    def log(message: str) -> None:
        if verbose:
            print(f"[hammer] {message}", file=sys.stderr, flush=True)

    hammer_dir = hammer_dir.resolve()
    source_bin = source_bin.resolve()
    studiomdl = source_bin / "studiomdl.exe"
    vtex = source_bin / "vtex.exe"
    if not studiomdl.is_file():
        raise NeoFormatError(f"studiomdl.exe was not found in {source_bin}")
    if not vtex.is_file():
        raise NeoFormatError(f"vtex.exe was not found in {source_bin}")
    game_dir = find_game_directory(source_bin, game_override)
    log(f"Source bin: {source_bin}")
    log(f"Game directory: {game_dir}")

    staged_materialsrc = game_dir / "materialsrc"
    game_materials = game_dir / "materials"
    game_models = game_dir / "models"
    staged_materialsrc.mkdir(parents=True, exist_ok=True)
    game_materials.mkdir(parents=True, exist_ok=True)
    game_models.mkdir(parents=True, exist_ok=True)

    for vmt in (hammer_dir / "materials").rglob("*.vmt"):
        relative = vmt.relative_to(hammer_dir / "materials")
        target = game_materials / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(vmt, target)
    resource_root = hammer_dir / "resource"
    if resource_root.is_dir():
        for resource in resource_root.rglob("*"):
            if not resource.is_file():
                continue
            target = game_dir / "resource" / resource.relative_to(resource_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(resource, target)

    ffmpeg = None
    if ffmpeg_override is not None:
        ffmpeg = str(ffmpeg_override.resolve())
        if not Path(ffmpeg).is_file():
            raise NeoFormatError(f"ffmpeg executable was not found: {ffmpeg}")
    else:
        ffmpeg = shutil.which("ffmpeg")

    texture_jobs: list[dict] = []
    source_root = hammer_dir / "materialsrc"
    manifest_file = hammer_dir / "hammer_manifest.json"
    active_materials = {
        item["material"] for item in json.loads(manifest_file.read_text(encoding="utf-8"))
    }
    # A decompiled-brush export may stage every NEO texture so it can be
    # selected manually in Hammer even if no draw command currently uses it.
    active_materials.update(vmt.stem.removesuffix("_brush") for vmt in
                            (hammer_dir / "materials").rglob("*.vmt"))
    images = sorted(
        image for image in source_root.rglob("*")
        if image.is_file() and image.stem in active_materials
    )
    for image_number, image in enumerate(images, 1):
        relative = image.relative_to(source_root)
        compiled_vtf = game_materials / relative.with_suffix(".vtf")
        if compiled_vtf.is_file() and not override_textures:
            log(f"Texture {image_number}/{len(images)}: cached {compiled_vtf.name}")
            texture_jobs.append({
                "source": str(image), "output": str(compiled_vtf),
                "exit_code": 0, "skipped_existing": True,
            })
            continue
        log(f"Texture {image_number}/{len(images)}: {image.name}")
        started = time.monotonic()
        staged = staged_materialsrc / relative.with_suffix(".tga")
        staged.parent.mkdir(parents=True, exist_ok=True)
        width = height = 0
        header = image.read_bytes()[:128]
        if image.suffix.lower() == ".dds" and header[:4] == b"DDS ":
            height, width = struct.unpack_from("<II", header, 12)
        elif image.suffix.lower() == ".tga" and len(header) >= 18:
            width, height = struct.unpack_from("<HH", header, 12)
        elif image.suffix.lower() == ".bmp" and len(header) >= 26 and header[:2] == b"BM":
            width, height = struct.unpack_from("<ii", header, 18)
            width, height = abs(width), abs(height)
        scale_filter = None
        if width and height:
            # Source 1's vtex requires power-of-two dimensions and can crash
            # outright for images such as 176x176 or 768x768. Reduce each
            # dimension independently to the largest permitted power of two.
            size_limit = max(1, max_vtf_size)
            target_width = 1 << (min(width, size_limit).bit_length() - 1)
            target_height = 1 << (min(height, size_limit).bit_length() - 1)
        else:
            target_width, target_height = width, height
        overview_image = relative.parts and relative.parts[0].lower() == "overviews"
        if overview_image and width and height and (width != height or radar_rotation != "none"):
            square = max(width, height)
            target_square = 1 << (min(square, max(1, max_vtf_size)).bit_length() - 1)
            overview_filters = []
            if radar_rotation == "cw":
                overview_filters.append("transpose=1")
            elif radar_rotation == "ccw":
                overview_filters.append("transpose=2")
            elif radar_rotation == "180":
                overview_filters.extend(("hflip", "vflip"))
            if width != height:
                overview_filters.append(
                    f"pad={square}:{square}:(ow-iw)/2:(oh-ih)/2:black"
                )
            overview_filters.append(f"scale={target_square}:{target_square}")
            scale_filter = ",".join(overview_filters)
            target_width = target_height = target_square
            rotation_message = (
                f", rotating {radar_rotation}" if radar_rotation != "none" else ""
            )
            log(
                f"  padding {width}x{height} to {square}x{square}"
                f"{rotation_message}, resizing to {target_square}x{target_square}"
            )
        elif target_width != width or target_height != height:
            scale_filter = f"scale={target_width}:{target_height}"
            log(f"  resizing {width}x{height} to {target_width}x{target_height}")

        if image.suffix.lower() == ".tga" and scale_filter is None:
            shutil.copyfile(image, staged)
            conversion = {"exit_code": 0, "command": ["copy", str(image), str(staged)]}
        elif ffmpeg:
            ffmpeg_command = [ffmpeg, "-y", "-i", str(image)]
            if scale_filter:
                ffmpeg_command.extend(["-vf", scale_filter])
            ffmpeg_command.extend(["-frames:v", "1", "-update", "1", str(staged)])
            conversion = run_source_tool(
                ffmpeg_command,
                hammer_dir,
                tool_timeout,
            )
        else:
            texture_jobs.append({
                "source": str(image), "exit_code": -1,
                "error": "image conversion requires ffmpeg; pass --ffmpeg or add it to PATH",
            })
            continue
        if conversion["exit_code"] != 0:
            texture_jobs.append({"source": str(image), "conversion": conversion, "exit_code": -1})
            log(f"  image conversion failed after {time.monotonic() - started:.1f}s")
            continue
        log(f"  running vtex (timeout {tool_timeout:g}s)")
        result = run_source_tool(
            [str(vtex), "-nopause", "-game", str(game_dir), str(staged)],
            source_bin, tool_timeout,
        )
        texture_jobs.append({"source": str(image), "staged": str(staged), **result})
        status = "timed out" if result.get("timed_out") else f"exit {result['exit_code']}"
        log(f"  vtex {status} after {time.monotonic() - started:.1f}s")

    model_jobs: list[dict] = []
    qcs = sorted((hammer_dir / "modelsrc").rglob("*.qc")) if compile_models else []
    if not compile_models:
        log("Skipping model compilation for textured-brush workflow")
    for qc_number, qc in enumerate(qcs, 1):
        qc = qc.resolve()
        qc_text = qc.read_text(encoding="utf-8", errors="replace")
        model_match = re.search(
            r'^\s*\$modelname\s+"([^"]+)"', qc_text,
            re.IGNORECASE | re.MULTILINE,
        )
        compiled_model = (
            game_models / Path(model_match.group(1).replace("\\", "/"))
            if model_match else None
        )
        if compiled_model is not None and compiled_model.is_file() and not override_models:
            log(f"Model {qc_number}/{len(qcs)}: cached {compiled_model.name}")
            model_jobs.append({
                "qc": str(qc), "output": str(compiled_model),
                "exit_code": 0, "skipped_existing": True,
            })
            continue
        log(f"Model {qc_number}/{len(qcs)}: {qc.name}")
        started = time.monotonic()
        result = run_source_tool(
            [str(studiomdl), "-game", str(game_dir), qc.name], qc.parent, tool_timeout
        )
        model_jobs.append({"qc": str(qc), **result})
        status = "timed out" if result.get("timed_out") else f"exit {result['exit_code']}"
        log(f"  studiomdl {status} after {time.monotonic() - started:.1f}s")

    return {
        "source_bin": str(source_bin),
        "game_dir": str(game_dir),
        "textures": texture_jobs,
        "models": model_jobs,
        "texture_failures": sum(job.get("exit_code") != 0 for job in texture_jobs),
        "model_failures": sum(job.get("exit_code") != 0 for job in model_jobs),
    }


def safe_texture_name(raw: bytes, fallback: str) -> str:
    name = raw.split(b"\0", 1)[0].decode("latin-1", errors="replace").strip()
    name = Path(name.replace("\\", "/")).name
    stem = Path(name).stem
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
    return cleaned or fallback


def write_tga(path: Path, width: int, height: int, pixel_size: int, pixels: bytes) -> None:
    if pixel_size == 4:
        image_type = 2
        depth = 32
        descriptor = 0x28  # top-left origin and 8 alpha bits
        converted = bytearray(len(pixels))
        for pos in range(0, len(pixels), 4):
            r, g, b, a = pixels[pos : pos + 4]
            converted[pos : pos + 4] = bytes((b, g, r, a))
        payload = converted
    elif pixel_size == 1:
        image_type = 3
        depth = 8
        descriptor = 0x20
        payload = pixels
    else:
        raise NeoFormatError(f"unsupported embedded texture pixel size: {pixel_size}")

    header = struct.pack(
        "<BBBHHBHHHHBB",
        0, 0, image_type, 0, 0, 0, 0, 0, width, height, depth, descriptor,
    )
    path.write_bytes(header + payload)


def extract_textures(neo: NeoFile, destination: Path, override: bool = False) -> list[dict]:
    blob = neo.lump_data(2)
    if len(blob) < 4:
        return []
    count = struct.unpack_from("<I", blob, 0)[0]
    table_end = 4 + count * 4
    if count > 100_000 or table_end > len(blob):
        raise NeoFormatError("invalid embedded texture offset table")
    offsets = list(struct.unpack_from(f"<{count}I", blob, 4)) if count else []
    destination.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for index, offset in enumerate(offsets):
        next_offset = offsets[index + 1] if index + 1 < count else len(blob)
        result = {"index": index, "offset": offset, "end": next_offset}
        if offset + 48 > len(blob) or next_offset <= offset:
            result["error"] = "invalid entry bounds"
            results.append(result)
            continue

        name = safe_texture_name(bytes(blob[offset : offset + 32]), f"texture_{index:04d}")
        width, height, pixel_size, texture_format = struct.unpack_from("<IIII", blob, offset + 32)
        required = width * height * pixel_size
        pixel_start = offset + 48
        available = max(0, next_offset - pixel_start)
        payload = bytes(blob[pixel_start:next_offset])
        reference = payload.split(b"\0", 1)[0].decode("latin-1", errors="replace").strip()
        reference = reference.replace("\\", "/").lstrip("/")
        result.update({
            "name": name,
            "width": width,
            "height": height,
            "pixel_size": pixel_size,
            "format": texture_format,
            "payload_bytes": available,
        })

        if reference.lower().endswith((".dds", ".tga", ".png")):
            source = neo.path.parent.parent / "tex" / Path(reference)
            if source.is_file():
                output = destination / f"{index:04d}_{name}{source.suffix.lower()}"
                if output.is_file() and not override:
                    result["skipped_existing"] = True
                else:
                    shutil.copyfile(source, output)
                result["external_reference"] = reference
                result["output"] = str(output)
                results.append(result)
                continue
            result["error"] = "external texture was not found"
            result["external_reference"] = reference
            results.append(result)
            continue

        if width == 0 or height == 0 or required > available or pixel_size not in (1, 4):
            result["error"] = "entry is compressed or uses an unsupported layout"
            results.append(result)
            continue

        output = destination / f"{index:04d}_{name}.tga"
        if output.is_file() and not override:
            result["skipped_existing"] = True
        else:
            write_tga(output, width, height, pixel_size, bytes(blob[pixel_start : pixel_start + required]))
        result["output"] = str(output)
        results.append(result)
    return results


def command_inspect(args: argparse.Namespace) -> int:
    neo = NeoFile.read(args.input)
    report = neo.report()
    print(json.dumps(report, indent=2))
    return 0


def command_export(args: argparse.Namespace) -> int:
    neo = NeoFile.read(args.input)
    override_textures = args.override in ("textures", "all")
    override_models = args.override in ("models", "all")
    decompiled_input = args.decompiled_vmf or args.decompiled_map
    if args.hammer and (args.blender_axes or args.source_axes):
        raise NeoFormatError("--hammer uses native Source coordinates and cannot be combined with an axis option")
    if args.source_bin and not args.hammer:
        raise NeoFormatError("--source-bin requires --hammer")
    if decompiled_input and not args.hammer:
        raise NeoFormatError("--decompiled-map/--decompiled-vmf requires --hammer")
    if args.transfer_lights and not decompiled_input:
        raise NeoFormatError("--transfer-lights requires --decompiled-map or --decompiled-vmf")
    if decompiled_input and not decompiled_input.is_file():
        raise NeoFormatError(f"decompiled map file was not found: {decompiled_input}")
    if args.map_bsp and not args.map_bsp.is_file():
        raise NeoFormatError(f"map BSP file was not found: {args.map_bsp}")
    if args.overview and not args.hammer:
        raise NeoFormatError("--overview requires --hammer")
    if args.overview and not args.overview.is_file():
        raise NeoFormatError(f"overview definition was not found: {args.overview}")
    if args.radar_rotate != "none" and not args.overview:
        raise NeoFormatError("--radar-rotate requires --overview")
    if args.radar_rotate != "none" and not args.source_bin:
        raise NeoFormatError("--radar-rotate requires --source-bin to compile the rotated image")
    if (args.game_dir or args.ffmpeg) and not args.source_bin:
        raise NeoFormatError("--game-dir and --ffmpeg require --source-bin")
    if args.tool_timeout <= 0:
        raise NeoFormatError("--tool-timeout must be greater than zero")
    if args.max_vtf_size <= 0:
        raise NeoFormatError("--max-vtf-size must be greater than zero")
    map_metadata = read_bsp_map_metadata(args.map_bsp) if args.map_bsp else {}
    sky_name = map_metadata.get("sky_name") or "sky_day01_01"
    args.output.mkdir(parents=True, exist_ok=True)
    report = neo.report()
    if map_metadata:
        report["map_bsp"] = {"path": str(args.map_bsp), **map_metadata}
    report["geometry"] = export_obj(
        neo,
        args.output / f"{args.input.stem}.obj",
        args.scale,
        args.source_axes,
        args.blender_axes,
        args.flip_v,
        args.split_objects,
        override_textures,
    )
    if not args.no_textures:
        report["textures"] = extract_textures(
            neo, args.output / "textures", override_textures,
        )
    if args.hammer:
        report["hammer"] = export_hammer(
            neo, args.output / "hammer", args.scale, args.flip_v,
            export_all_textures=bool(decompiled_input),
            vmf_name=f"{args.input.stem}_models" if decompiled_input else None,
            override=override_textures,
            target_game=args.target_game,
            sky_name=sky_name,
        )
        if decompiled_input:
            report["textured_brushes"] = export_decompiled_vmf(
                neo, decompiled_input,
                args.output / "hammer" / f"{args.input.stem}.vmf",
                args.scale,
                args.max_vtf_size,
                args.target_game,
                args.transfer_lights,
                sky_name,
            )
        if args.map_bsp and map_metadata.get("sky_name"):
            report["skybox"] = export_bsp_skybox(
                args.map_bsp, args.output / "hammer", sky_name,
                override_textures,
            )
        if args.overview:
            report["overview"] = export_css_overview(
                neo, args.overview, args.output / "hammer", args.scale,
                args.input.stem, args.radar_rotate,
            )
        if args.source_bin:
            report["hammer_compile"] = compile_hammer_output(
                args.output / "hammer", args.source_bin, args.game_dir, args.ffmpeg,
                args.tool_timeout, args.verbose, args.max_vtf_size,
                override_textures=override_textures,
                override_models=override_models,
                radar_rotation=args.radar_rotate,
            )
    if args.map_bsp:
        location_dir = args.output / "hammer" if args.hammer else args.output
        report["locations"] = export_bsp_location_names(
            args.map_bsp,
            location_dir / f"{args.input.stem}_locations.json",
        )
    report_path = args.output / f"{args.input.stem}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), **report["geometry"]}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="validate a NEO file and print its lump table")
    inspect_parser.add_argument("input", type=Path)
    inspect_parser.set_defaults(func=command_inspect)

    export_parser = subparsers.add_parser("export", help="export OBJ geometry and embedded TGA textures")
    export_parser.add_argument("input", type=Path)
    export_parser.add_argument("output", type=Path)
    export_parser.add_argument("--scale", type=float, default=1.0, help="vertex scale (default: 1.0)")
    axes = export_parser.add_mutually_exclusive_group()
    axes.add_argument(
        "--source-axes",
        action="store_true",
        help="convert (x,y,z) to (x,z,-y); leave off for GoldSrc/Source coordinates",
    )
    axes.add_argument(
        "--blender-axes",
        action="store_true",
        help="orient for Blender with a clockwise X rotation and X mirror while preserving outward winding",
    )
    export_parser.add_argument(
        "--flip-v",
        dest="flip_v",
        action="store_true",
        help="vertically flip texture coordinates (default)",
    )
    export_parser.add_argument(
        "--no-flip-v",
        dest="flip_v",
        action="store_false",
        help="preserve raw NEO texture-coordinate V values",
    )
    export_parser.set_defaults(flip_v=True)
    export_parser.add_argument(
        "--split-objects",
        action="store_true",
        help="import each NEO draw mesh as a separately selectable Blender object",
    )
    export_parser.add_argument("--no-textures", action="store_true")
    export_parser.add_argument(
        "--override",
        nargs="?",
        const="all",
        choices=("models", "textures", "all"),
        help=(
            "override cached models, textures/VTFs, or all assets; "
            "using --override without a value means all"
        ),
    )
    export_parser.add_argument(
        "--hammer",
        action="store_true",
        help="also generate Source SMD/QC/VMT files and a prop_static VMF scaffold",
    )
    export_parser.add_argument(
        "--target-game",
        choices=("gmod", "css"),
        default="gmod",
        help="select target-specific entity mappings (default: gmod)",
    )
    decompiled_group = export_parser.add_mutually_exclusive_group()
    decompiled_group.add_argument(
        "--decompiled-map",
        type=Path,
        metavar="PATH",
        help="create a textured Source brush VMF from a decompiled GoldSrc .map using NEO geometry matching",
    )
    decompiled_group.add_argument(
        "--decompiled-vmf",
        type=Path,
        metavar="PATH",
        help="create a textured Source brush VMF from a J.A.C.K-converted VMF using NEO geometry matching",
    )
    export_parser.add_argument(
        "--transfer-lights",
        action="store_true",
        help="add positive NEO renderer-light records as Source light entities to the decompiled brush VMF",
    )
    export_parser.add_argument(
        "--map-bsp", "--location-bsp",
        dest="map_bsp",
        type=Path,
        metavar="PATH",
        help=(
            "read reusable metadata from the original GoldSrc BSP, including "
            "locations and skybox; --location-bsp is a compatibility alias"
        ),
    )
    export_parser.add_argument(
        "--overview",
        type=Path,
        metavar="PATH",
        help="convert a GoldSrc overview TXT/BMP into staged Counter-Strike: Source radar assets",
    )
    export_parser.add_argument(
        "--radar-rotate",
        choices=("none", "cw", "ccw", "180"),
        default="none",
        help="optionally rotate only the compiled radar image (default: none)",
    )
    export_parser.add_argument(
        "--source-bin",
        type=Path,
        metavar="PATH",
        help="Source game bin folder containing studiomdl.exe and vtex.exe; implies auto-compilation with --hammer",
    )
    export_parser.add_argument(
        "--game-dir",
        type=Path,
        metavar="PATH",
        help="game folder containing gameinfo.txt (normally inferred from --source-bin)",
    )
    export_parser.add_argument(
        "--ffmpeg",
        type=Path,
        metavar="EXE",
        help="optional ffmpeg executable used to convert external DDS textures before vtex",
    )
    export_parser.add_argument(
        "--tool-timeout",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="maximum time for each ffmpeg, vtex, or studiomdl invocation (default: 60)",
    )
    export_parser.add_argument(
        "--verbose",
        action="store_true",
        help="print live Hammer conversion progress and per-tool completion status",
    )
    export_parser.add_argument(
        "--max-vtf-size",
        type=int,
        default=1024,
        metavar="PIXELS",
        help="resize textures down to Source-compatible power-of-two dimensions (default limit: 1024)",
    )
    export_parser.set_defaults(func=command_export)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, NeoFormatError, struct.error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
