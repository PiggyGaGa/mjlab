# Apartment collision approximation

`collision.xml` approximates `../scene_transform_p5.ply` in the same right-handed,
Z-up, metre coordinate system. The floor top is at z=0. The ground slab spans
x=[-2.8, 4.1] m and y=[-6.1, 3.8] m. This rectangular support slab also extends
outside the irregular wall outline; the voxel surfaces define the room boundary.
The default spawn at (0, 0, 0) is inside the open central area.

The model contains a ground slab and 36 static surface boxes. Merging 20,468
occupied 10 cm voxels produces 89 boxes; pruning removes 53 small fragments.
The retained boxes follow the walls, right-side recesses, ceiling, and any
sufficiently dense object surfaces. These hidden collision proxies use visualization
group 3. Two opaque group 1 pillars provide camera-visible obstacles with matching
collision geometry. Enable collision visibility in the viewer to inspect alignment.

Rebuild after editing the PLY:

```sh
uv run src/mjlab/asset_zoo/3dgs/apartment/collision/generate_collision.py
```

The generator also updates the manifest's visual path.
It keeps Gaussian centres with opacity >=0.5 (source opacity is stored as logits),
uses padded 0.1%–99.9% coordinate quantiles to reject distant outliers, and requires
at least three centres per voxel. Six-connected components smaller than eight
voxels are discarded. Points below z=0.1 are replaced by the floor slab.

The merger recursively partitions occupied cells, choosing axis-aligned splits
that minimize the sum of the children's bounding volumes. A region becomes one
box when at least 20% of its cells were occupied and every added cell is within
40 cm of an original occupied cell centre. All original occupied cells remain
covered by the merger before pruning. This fills small scan gaps and smooths
uneven wall thickness, allowing
large wall and ceiling boxes instead of fragmented voxel columns. It can thicken
surfaces or narrow openings; gaps up to 80 cm across may close. A doorway
regression verifies that a one-metre opening retains its central passage. Reduce
`MAX_SURFACE_EXPANSION` or increase `MIN_BOX_FILL` in the generator if tighter
geometric fidelity matters more than the number of collision geoms.

After merging, surface boxes with full volume below 0.1 m³ are discarded.
MJCF `size` values are half-extents, so volume is `8 * size_x * size_y * size_z`.
Boxes exactly at the threshold are retained. Change `MIN_SURFACE_BOX_VOLUME`
in the generator to adjust this cutoff. Pruning may remove small real surfaces
and leave local collision gaps; the final model no longer covers every source
voxel. The ground slab and explicit pillars are exempt from this filter.

This is a surface approximation, not a reconstruction of solid furniture or a
semantic floor plan. The supplied cloud has no prominent interior wall or
furniture clusters at walking height after filtering; no such obstacles are
invented. Missing/transparent surfaces, thin furniture legs, small objects, and
details below 10 cm may lack collisions. Bounds and Gaussian centres do not
capture the full extent of large splats. Inspect the collision overlay before
using this approximation for contact-sensitive training; edit or replace it with
measured geometry where greater accuracy is needed. Fewer geoms reduce collision
scene complexity; the actual simulation speedup depends on active contacts and
the simulation backend and has not been benchmarked.
