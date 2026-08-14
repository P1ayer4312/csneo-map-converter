# CS NEO Map Converter

Experimental converter for the `.neo` rendering maps included with Counter-Strike: NEO. It can export geometry and textures for Blender and generate an editable Source 1/Hammer starting point from a decompiled GoldSrc map.

The result is a reconstruction, not a finished port. Incorrect or low-confidence materials, gameplay logic, lighting, visibility and map optimization may require manual work in Hammer.

## Dependencies

- Python 3.10 or newer (no third-party Python packages)
- Original `.neo` files and, where required, their matching `.bsp` files
- A GoldSrc BSP-to-MAP decompiler for editable brush conversion
- Blender (optional, for inspecting OBJ exports)
- Source 1 game tools: `vtex.exe` and `studiomdl.exe` (optional automatic Hammer asset compilation)
- FFmpeg (optional, but required to compile external DDS textures automatically)

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
    --location-bsp linux\czero\maps\neo_00collision.bsp
```

`--decompiled-map` and `--decompiled-vmf` are mutually exclusive.

Use `--ffmpeg C:\path\to\ffmpeg.exe` if FFmpeg is not available through `PATH`.

## Output

Depending on the selected options, the output contains:

- OBJ/MTL geometry and extracted textures
- Source SMD, QC, VMT and VMF files under `hammer/`
- A confidence report for automatically assigned brush materials
- An entity manual-review report
- A UTF-8 location JSON containing recovered CP932/Shift-JIS names and IDs

Recognized entities and triggers are translated where practical. Unknown brush entities are retained as non-solid manual-review helpers. NEO `PLACE_NAME` volumes are converted to non-solid `trigger_multiple` entities.

Run `python neo_map_converter.py export --help` for every available option.
