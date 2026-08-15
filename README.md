# CS NEO Map Converter

Experimental converter for the `.neo` rendering maps included with Counter-Strike: NEO. It can export geometry and textures for Blender and generate an editable Source 1/Hammer starting point from a decompiled GoldSrc map.

The result is a reconstruction, not a finished port. Incorrect or low-confidence materials, gameplay logic, lighting, visibility and map optimization may require manual work in Hammer.

## NEO format overview

A `.neo` file is a little-endian rendering-data container rather than a complete
GoldSrc map. Collision brushes, gameplay entities and compiled BSP lighting live
in the matching `.bsp`; this is why editable Hammer conversion also needs a
decompiled MAP or VMF.

The file begins with a 132-byte header:

- `uint32` format version; versions 1014 through 1017 are currently recognized
- Sixteen `(uint32 offset, uint32 length)` lump descriptors
- Lump offsets are absolute file offsets and each populated lump follows the header

The sixteen lumps are:

| Index | Contents                                | Record layout currently used                                |
| ----: | --------------------------------------- | ----------------------------------------------------------- |
|     0 | Vertex positions                        | Three `float32` values (12 bytes)                           |
|     1 | Vertex colors/additional vertex stream  | Four `float32` values (16 bytes)                            |
|     2 | Texture table and image data/references | Variable length                                             |
|     3 | Scene nodes                             | 40-byte records                                             |
|     4 | Texture information                     | UV start, texture index and flags (12 bytes)                |
|     5 | Mesh records                            | 36 signed integers (144 bytes)                              |
|     6 | Vertex indices                          | `uint32` (4 bytes)                                          |
|     7 | Reserved/unknown                        | Variable length                                             |
|     8 | Draw commands                           | OpenGL mode, start and signed count (12 bytes)              |
|     9 | Lights                                  | 84-byte records                                             |
|    10 | Texture coordinates                     | Two `float32` values (8 bytes)                              |
|    11 | Mesh attributes                         | 80-byte records                                             |
|    12 | Effect shader names                     | 132-byte records                                            |
|    13 | Effect shader properties                | Count/offset table followed by variable records; texture properties use 136-byte records |
|    14 | Reserved/unknown                        | Variable length                                             |
|    15 | Models                                  | 48-byte records                                             |

The texture lump starts with a texture count and an offset table. Each texture
entry contains a 32-byte name, width, height, pixel size, format and payload.
The payload is either embedded pixel data or a path to an external DDS, TGA or
PNG under the game files.

Mesh records reference draw-command IDs and texture information. Draw commands
store OpenGL primitive types such as triangles, strips, fans, quads and polygons.
A negative draw count means that the command uses the index lump. The converter
triangulates these primitives and pairs the independent vertex and UV streams
when writing OBJ/SMD geometry. The first three fields of each mesh-attribute
record are the shader ID, effect-property start, and effect-property count.
Shader-driven meshes may obtain their diffuse texture through those properties.

Each 84-byte light record contains a position (`float3`), signed intensity,
three RGBA color groups (primary/diffuse, secondary and ambient), and five
reserved values. With `--transfer-lights`, positive records become Source point
lights. Source has no dependable equivalent for NEO's negative/subtractive
lights, so those are retained in the light-review report for manual handling.

The optimized `OPT_LightmappedDiffuse2Cg.fx` shader batches as many as three
diffuse materials into one draw command. Its `TexCoord0` is a padded float4:
`xy` contains the ordinary UV while `z` selects `diffuseMap0`, `diffuseMap1`, or
`diffuseMap2` per vertex. Each vertex stores this float4 as an interleaved pair
of vec2 records in lump 10 (`xy`, followed by `z` and padding). The converter
reads this selector and emits separate OBJ/SMD
material assignments per triangle. When no diffuse texture is serialized, the
secondary/lightmap texture may still be used as a visible fallback.

This layout was reconstructed from the game binaries and observed files. Some
record fields and the reserved lumps remain unknown.

## Dependencies

- Python 3.10 or newer (no third-party Python packages)
- Original `.neo` files and, where required, their matching `.bsp` files
- A GoldSrc BSP-to-MAP decompiler for editable brush conversion
- Blender (optional, for inspecting OBJ exports)
- Source 1 game tools: `vtex.exe` and `studiomdl.exe` (optional automatic Hammer asset compilation)
- FFmpeg (optional, but required to compile external DDS textures automatically;
  when available on `PATH`, it also creates real-alpha Blender textures for
  additive sprites whose original DDS uses an opaque black background)

## Basic usage

Inspect a NEO file:

```powershell
python neo_map_converter.py inspect linux\czero\maps\neo_02collision.neo
```

Export OBJ geometry and textures for Blender:

```powershell
python neo_map_converter.py export `
    linux\czero\maps\neo_02collision.neo `
    converted\neo_02collision_blender `
    --blender-axes --split-objects
```

Generate a textured brush VMF from a decompiled map and compile its materials for Garry's Mod:

```powershell
python neo_map_converter.py export `
    linux\czero\maps\neo_02collision.neo `
    converted\neo_02collision_gmod `
    --hammer `
    --decompiled-map decompiled\neo_02collision.map `
    --location-bsp linux\czero\maps\neo_02collision.bsp `
    --source-bin "C:\Program Files (x86)\Steam\steamapps\common\GarrysMod\bin" `
    --max-vtf-size 512 `
    --verbose
```

If the decompiled MAP was converted with J.A.C.K, provide its VMF instead:

```powershell
python neo_map_converter.py export `
    linux\czero\maps\neo_00collision.neo `
    converted\neo_00collision_gmod `
    --hammer `
    --decompiled-vmf decompiled\neo_00collision.vmf `
    --transfer-lights `
    --location-bsp linux\czero\maps\neo_00collision.bsp
```

`--decompiled-map` and `--decompiled-vmf` are mutually exclusive.

For Counter-Strike: Source, add `--target-game css`. This converts GoldSrc
`info_player_start`/`info_player_deathmatch` entities into native CS:S
counter-terrorist/terrorist spawns. The default target remains `gmod`.

```powershell
python neo_map_converter.py export `
    linux\czero\maps\neo_06collision.neo `
    converted\neo_06collision_css `
    --hammer --target-game css `
    --decompiled-vmf decompiled\neo_06collision.vmf `
    --location-bsp linux\czero\maps\neo_06collision.bsp `
    --source-bin "C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Source\bin" `
    --max-vtf-size 512 --verbose
```

Use `--ffmpeg C:\path\to\ffmpeg.exe` if FFmpeg is not available through `PATH`.

## Output

Depending on the selected options, the output contains:

- OBJ/MTL geometry and extracted textures
- Source SMD, QC, VMT and VMF files under `hammer/`
- A confidence report for automatically assigned brush materials
- An entity manual-review report
- A UTF-8 location JSON containing recovered CP932/Shift-JIS names and IDs

OBJ material export detects embedded RGBA and alpha-capable DDS/TGA/PNG files
and writes `map_d` entries so Blender imports their transparent regions instead
of displaying the transparent background as solid black.

Recognized entities and triggers are translated where practical. Unknown brush entities are retained as non-solid manual-review helpers. NEO `PLACE_NAME` volumes are converted to non-solid `trigger_multiple` entities.

## CLI reference

General syntax:

```text
python neo_map_converter.py inspect INPUT
python neo_map_converter.py export INPUT OUTPUT [options]
```

### `inspect`

- `INPUT`: `.neo` file to validate and describe.
- `-h`, `--help`: Show command help.

`inspect` prints the file version and offset, length, record size and count of
every lump without exporting anything.

### `export` arguments and flags

- `INPUT`: Source `.neo` file.
- `OUTPUT`: Directory in which the conversion is created.
- `--scale NUMBER`: Multiply exported coordinates by this value; default is `1.0`.
- `--source-axes`: Export OBJ coordinates as `(x, z, -y)`.
- `--blender-axes`: Orient for Blender with a 90-degree clockwise X rotation and X mirror while preserving outward face winding.
- `--flip-v`: Flip texture-coordinate V values; this is the default.
- `--no-flip-v`: Preserve the raw NEO V values.
- `--split-objects`: Write each draw command as a separately selectable OBJ object.
- `--no-textures`: Skip the ordinary top-level texture extraction pass.
- `--hammer`: Also create Source SMD, QC, VMT and VMF files under `OUTPUT/hammer`.
- `--target-game gmod`: Use the default Garry's Mod-compatible entity mappings.
- `--target-game css`: Use Counter-Strike: Source mappings, including native T/CT spawn classes.
- `--decompiled-map PATH`: Rebuild editable Source brushes and entities from a decompiled GoldSrc MAP, then assign NEO materials by geometric matching.
- `--decompiled-vmf PATH`: Use a J.A.C.K-converted VMF as the brush/entity source instead of a MAP.
- `--transfer-lights`: Add positive records from NEO lump 9 as Source `light` entities to the decompiled brush VMF. Subtractive/invalid records and clamped extreme intensities are listed in `*_light_review.json`.
- `--location-bsp PATH`: Decode `PLACE_NAME` labels and IDs from the original BSP's CP932 entity data and export a UTF-8 location JSON.
- `--source-bin PATH`: Locate `vtex.exe` and `studiomdl.exe` and automatically compile generated textures/models for the target Source game.
- `--game-dir PATH`: Override the game directory containing `gameinfo.txt`; requires `--source-bin`.
- `--ffmpeg EXE`: Select FFmpeg explicitly for DDS conversion; requires `--source-bin`.
- `--tool-timeout SECONDS`: Per-process timeout for FFmpeg, VTex and StudioMDL; default is 60 seconds.
- `--verbose`: Print live asset-conversion progress and tool status.
- `--max-vtf-size PIXELS`: Downscale compiled textures to a Source-compatible power-of-two limit; default is 1024.
- `-h`, `--help`: Show command help.

`--source-axes` and `--blender-axes` are mutually exclusive. So are
`--decompiled-map` and `--decompiled-vmf`. Decompiled brush conversion requires
`--hammer`; `--transfer-lights` additionally requires a decompiled MAP or VMF;
`--game-dir` and `--ffmpeg` require `--source-bin`.

Run `python neo_map_converter.py export --help` for every available option.
