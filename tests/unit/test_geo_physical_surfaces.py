"""The ``.geo`` export must put each physical surface on the face it names.

The physical surface names are the object ids of the ``.3dm`` next to the
``.geo``: the frontend shows that ``.3dm``, users assign materials to its
objects, and the simulation methods apply each material to the surfaces of
the physical surface with that object's id. A physical surface on another
face puts the material on the wrong wall, without any error.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from geometry_pipeline import process_geometry
from geometry_pipeline.core.ir import Face, Mesh, Vertex
from geometry_pipeline.io.exporters.mesh_geo import GmshGeoExporter
from geometry_pipeline.io.exporters.mesh_three_dm import MeshThreeDMExporter
from geometry_pipeline.io.importers.obj import ObjImporter

CUBE_VERTS = [
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
]
CUBE_FACES = [[1, 4, 3, 2], [5, 6, 7, 8], [1, 2, 6, 5], [3, 4, 8, 7], [2, 3, 7, 6], [1, 5, 8, 4]]


def faces_on_wrong_surface(geo: Path) -> list[str]:
    """Object ids whose physical surface is not the face they are in the sibling ``.3dm``."""
    import gmsh
    import rhino3dm

    def corners(points) -> set:
        return {tuple(np.round(p, 6)) for p in points}

    doc = rhino3dm.File3dm.Read(str(geo.with_suffix(".3dm")))
    objects = {}
    for obj in doc.Objects:
        v = obj.Geometry.Vertices
        objects[str(obj.Attributes.Id)] = corners([v[i].X, v[i].Y, v[i].Z] for i in range(len(v)))

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(geo))
        surfaces = {
            tag: corners(
                gmsh.model.getValue(0, point, [])
                for _, point in gmsh.model.getBoundary([(2, tag)], recursive=True)
            )
            for _, tag in gmsh.model.getEntities(2)
        }
    finally:
        gmsh.finalize()

    groups = re.findall(r'Physical Surface\("([^"]+)"\) = \{ (\d+) \}', geo.read_text())
    return [name for name, tag in groups if surfaces.get(int(tag)) != objects[name]]


def test_geo_export_physical_surfaces_with_several_materials(tmp_path):
    # Materials not sorted by name, like the backend's example Measurement Room.
    materials = ["M_3", "M_1", "M_2", "M_1", "M_1", "M_1"]
    mesh = Mesh(
        vertices=[Vertex(*v) for v in CUBE_VERTS],
        faces=[
            Face(vertex_indices=list(f), group="default", material=m)
            for f, m in zip(CUBE_FACES, materials)
        ],
    )
    # Written like the repaired bundle: the .3dm first, then the .geo reading it.
    MeshThreeDMExporter().write(mesh, tmp_path / "room.3dm")
    GmshGeoExporter(detect_cavities=False).write(mesh, tmp_path / "room.geo")

    assert faces_on_wrong_surface(tmp_path / "room.geo") == []


def test_inspect_geo_physical_surfaces_after_removed_face(tmp_path):
    # A zero-area face before the cube's faces: the pipeline removes it before
    # writing the inspect .geo, but the uploaded model's .3dm still has it.
    lines = [f"v {x} {y} {z}" for x, y, z in CUBE_VERTS] + ["v 0.5 0 0", "usemtl wall", "f 1 9 2"]
    lines += ["f " + " ".join(map(str, f)) for f in CUBE_FACES]
    obj = tmp_path / "room.obj"
    obj.write_text("\n".join(lines) + "\n")
    # The backend converts the upload to a .3dm with one object per face, in OBJ order.
    MeshThreeDMExporter().write(ObjImporter().load(obj), tmp_path / "room.3dm")

    # Same folder for upload and output, as in the backend.
    process_geometry(obj, tmp_path, detect_cavities=False)

    assert faces_on_wrong_surface(tmp_path / "room.geo") == []


def cube_mesh(verts, faces) -> Mesh:
    return Mesh(
        vertices=[Vertex(*v) for v in verts],
        faces=[Face(vertex_indices=list(f), group="default", material="wall") for f in faces],
    )


def object_ids(path: Path) -> list[str]:
    import rhino3dm

    return [str(obj.Attributes.Id) for obj in rhino3dm.File3dm.Read(str(path)).Objects]


def physical_surface_names(geo: Path) -> list[str]:
    return re.findall(r'Physical Surface\("([^"]+)"\)', geo.read_text())


def test_geo_export_matches_surface_with_t_junction_vertex(tmp_path):
    # The T-junction fix inserts (0.5, 0, 0) into the bottom and front outlines;
    # the uploaded model's .3dm has them without it.
    MeshThreeDMExporter().write(cube_mesh(CUBE_VERTS, CUBE_FACES), tmp_path / "room.3dm")
    faces = [list(f) for f in CUBE_FACES]
    faces[0] = [1, 4, 3, 2, 9]
    faces[2] = [1, 9, 2, 6, 5]
    mesh = cube_mesh([*CUBE_VERTS, (0.5, 0, 0)], faces)
    GmshGeoExporter(detect_cavities=False).write(mesh, tmp_path / "room.geo")

    assert physical_surface_names(tmp_path / "room.geo") == object_ids(tmp_path / "room.3dm")


def test_geo_export_matches_corner_moved_by_vertex_dedup(tmp_path):
    # Dedup can move a corner by up to vertex_merge (1 cm) per axis.
    MeshThreeDMExporter().write(cube_mesh(CUBE_VERTS, CUBE_FACES), tmp_path / "room.3dm")
    verts = list(CUBE_VERTS)
    verts[6] = (1.009, 0.991, 1.009)
    GmshGeoExporter(detect_cavities=False).write(
        cube_mesh(verts, CUBE_FACES), tmp_path / "room.geo"
    )

    assert physical_surface_names(tmp_path / "room.geo") == object_ids(tmp_path / "room.3dm")


def test_geo_export_names_surface_without_object_as_unmatched(tmp_path, caplog):
    # A surface the .3dm doesn't have: a triangle away from the cube.
    MeshThreeDMExporter().write(cube_mesh(CUBE_VERTS, CUBE_FACES), tmp_path / "room.3dm")
    verts = [*CUBE_VERTS, (3, 0, 0), (4, 0, 0), (3, 1, 0)]
    mesh = cube_mesh(verts, [*CUBE_FACES, [9, 10, 11]])
    GmshGeoExporter(detect_cavities=False).write(mesh, tmp_path / "room.geo")

    ids = object_ids(tmp_path / "room.3dm")
    assert physical_surface_names(tmp_path / "room.geo") == [*ids, "unmatched_surface_7"]
    assert "No object in" in caplog.text
