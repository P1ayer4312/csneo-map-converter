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

| Index | Contents                                | Record layout currently used                                                             |
| ----: | --------------------------------------- | ---------------------------------------------------------------------------------------- |
|     0 | Vertex positions                        | Three `float32` values (12 bytes)                                                        |
|     1 | Vertex colors/additional vertex stream  | Four `float32` values (16 bytes)                                                         |
|     2 | Texture table and image data/references | Variable length                                                                          |
|     3 | Scene nodes                             | 40-byte records                                                                          |
|     4 | Texture information                     | UV start, texture index and flags (12 bytes)                                             |
|     5 | Mesh records                            | 36 signed integers (144 bytes)                                                           |
|     6 | Vertex indices                          | `uint32` (4 bytes)                                                                       |
|     7 | Reserved/unknown                        | Variable length                                                                          |
|     8 | Draw commands                           | OpenGL mode, start and signed count (12 bytes)                                           |
|     9 | Lights                                  | 84-byte records                                                                          |
|    10 | Texture coordinates                     | Two `float32` values (8 bytes)                                                           |
|    11 | Mesh attributes                         | 80-byte records                                                                          |
|    12 | Effect shader names                     | 132-byte records                                                                         |
|    13 | Effect shader properties                | Count/offset table followed by variable records; texture properties use 136-byte records |
|    14 | Reserved/unknown                        | Variable length                                                                          |
|    15 | Models                                  | 48-byte records                                                                          |

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
