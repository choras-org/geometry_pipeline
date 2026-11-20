"""Shared repair helpers (C3 target).

Start with `room_center_from_mesh` (was `mesh_helpers.py`) so callers
can import a stable `_common` module. Additional shared helpers will be
moved here as the refactor proceeds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geometry_pipeline.core.ir import Mesh
from typing import Any, Dict, List, Tuple

import numpy as np

from geometry_pipeline.core.ir import Face
from geometry_pipeline.geometry_math.geometry_math import (
    compute_face_unit_normal,
    distance,
    newell_normal_from_points,
    offset_point_along_vector,
    polygon_centroid,
)


def _vertex_mean(vertices: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    n = len(vertices)
    return (
        sum(p[0] for p in vertices) / n,
        sum(p[1] for p in vertices) / n,
        sum(p[2] for p in vertices) / n,
    )


def _enclosed_volume_centroid(
    vertices: List[Tuple[float, float, float]],
    faces,
) -> Tuple[float, float, float] | None:
    """Centre of mass of the solid enclosed by a closed, consistently
    oriented surface.

    Uses the divergence-theorem decomposition into tetrahedra
    ``(origin, a, b, c)`` (polygon faces are fan-triangulated). A global
    orientation flip cancels in the ``moment / volume`` ratio, so only the
    *relative* consistency of the faces matters, not whether they all point
    in or out. Returns ``None`` when the mesh has no enclosed volume
    (open / planar / zero-volume), in which case the caller should fall back
    to a simpler estimate.
    """
    total_v = 0.0
    cx = cy = cz = 0.0
    for f in faces:
        vids = f.vertex_indices
        if len(vids) < 3:
            continue
        a = vertices[vids[0] - 1]
        for k in range(1, len(vids) - 1):
            b = vertices[vids[k] - 1]
            c = vertices[vids[k + 1] - 1]
            # 6 x signed volume of tetrahedron (origin, a, b, c)
            vol6 = (
                a[0] * (b[1] * c[2] - b[2] * c[1])
                - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0])
            )
            total_v += vol6
            # tetra centroid is (origin + a + b + c) / 4
            cx += vol6 * (a[0] + b[0] + c[0])
            cy += vol6 * (a[1] + b[1] + c[1])
            cz += vol6 * (a[2] + b[2] + c[2])

    if abs(total_v) < 1e-15:
        return None
    # the constant 1/6 (volume) and 1/4 (centroid) factors cancel except for
    # the 1/4 on the moment, which survives as (.../4) / (total_v) -> /4.
    inv = 1.0 / (4.0 * total_v)
    return (cx * inv, cy * inv, cz * inv)


def room_center_from_mesh(mesh: "Mesh") -> tuple[float, float, float]:
    """A reference point inside the room.

    Several repairs ('which side of this plane is the room?', 'do the normals
    point outward?') need an interior reference point. The naive mean of the
    vertices is biased by vertex density and, for non-convex rooms, is easily
    pulled outside the solid.

    We therefore prefer the **centre of mass of the enclosed volume**
    (divergence-theorem centroid), which ignores vertex-density bias and lies
    inside the solid for any star-convex room. Two safety nets:

    * if the mesh encloses no volume (open / planar surface) we fall back to
      the vertex mean;
    * if the computed centroid falls outside the vertex bounding box -- the
      tell-tale sign of an *inconsistently* oriented surface, for which the
      signed-volume integral is meaningless -- we also fall back to the
      vertex mean.

    Note: for strongly non-convex rooms (e.g. an L- or U-shaped plan) no
    single centroid is guaranteed to sit inside the material; a guaranteed
    interior point would require a heavier method (voxel sampling / pole of
    inaccessibility). The volume centroid is a strict improvement over the
    vertex mean and is sufficient for the orientation/PLC heuristics here.
    """
    if not mesh.vertices:
        return (0.0, 0.0, 0.0)

    vertices = [(v.x, v.y, v.z) for v in mesh.vertices]
    centroid = _enclosed_volume_centroid(vertices, mesh.faces)

    if centroid is not None:
        min_x = min(p[0] for p in vertices)
        min_y = min(p[1] for p in vertices)
        min_z = min(p[2] for p in vertices)
        max_x = max(p[0] for p in vertices)
        max_y = max(p[1] for p in vertices)
        max_z = max(p[2] for p in vertices)
        if (
            min_x <= centroid[0] <= max_x
            and min_y <= centroid[1] <= max_y
            and min_z <= centroid[2] <= max_z
        ):
            return centroid

    return _vertex_mean(vertices)


def orient_faces_outward(
    faces: List[Face],
    points: List[Tuple[float, float, float]],
    logger=None,
) -> Dict[str, Any]:
    """Turn faces around where needed so that every normal points out of the
    room's volume.

    Faces that point the wrong way are reversed in place. Vertex indices are
    1-based into ``points``.

    This works in two steps:

    1. Make neighbouring faces agree. Two faces that share an edge must walk
       along it in opposite directions. Starting from one face, we visit its
       neighbours and reverse any that disagree. Faces that are connected this
       way form a "piece" (e.g. the room, or a column standing in it).
    2. Decide which way each piece should face, and reverse the whole piece
       if needed:

       - The biggest closed piece is the room. Its normals should point out
         of the room.
       - A closed piece inside the room is an obstacle. Its normals should
         point into the obstacle (out of the room's volume).
       - Any other closed piece gets normals pointing out of it.
       - A piece with holes in it has no inside or outside, so we flip it if
         most of its area faces the room centre.

    Returns a diagnostics dict with the number of pieces, closed pieces and
    reversed faces.
    """
    P = np.asarray(points, dtype=float)
    # Move the origin to the middle of the model, so the volume sums below
    # don't lose precision on models far from the origin.
    P = P - P.mean(axis=0)
    flipped = [False] * len(faces)

    def edges(f):
        v = f.vertex_indices
        return [(a, b) for a, b in zip(v, v[1:] + v[:1]) if a != b]

    def tris(fi):
        # Split the face into triangles that all share its first corner. This
        # gives the right area and volume even for faces that aren't convex.
        p = P[np.asarray(faces[fi].vertex_indices) - 1]
        return p[0], p[1:-1], p[2:]

    def reverse(fi):
        faces[fi].vertex_indices.reverse()
        flipped[fi] = not flipped[fi]

    edge_faces: Dict[frozenset, List[int]] = {}
    for fi, f in enumerate(faces):
        for a, b in edges(f):
            edge_faces.setdefault(frozenset((a, b)), []).append(fi)

    # Step 1: walk from face to face and make neighbours agree. We only cross
    # edges shared by exactly two faces. An edge with only one face is a hole,
    # and an edge with three or more faces has no clear neighbour.
    # Each face is only looked at once. If the model is broken and two
    # neighbours want different directions, whichever came first wins.
    piece = [-1] * len(faces)
    pieces: List[List[int]] = []
    for start in range(len(faces)):
        if piece[start] >= 0:
            continue
        piece[start] = len(pieces)
        members, stack = [start], [start]
        while stack:
            fi = stack.pop()
            for a, b in edges(faces[fi]):
                nbrs = edge_faces[frozenset((a, b))]
                if len(nbrs) != 2:
                    continue
                nj = nbrs[1] if nbrs[0] == fi else nbrs[0]
                if piece[nj] >= 0:
                    continue
                if (a, b) in edges(faces[nj]):
                    reverse(nj)
                piece[nj] = piece[fi]
                members.append(nj)
                stack.append(nj)
        pieces.append(members)

    def flip(members):
        for fi in members:
            reverse(fi)

    # The volume is positive if the normals point outward and negative if they
    # point inward. The centroid is the centre of mass of the enclosed space.
    def volume_and_centroid(members):
        vol, moment = 0.0, np.zeros(3)
        for a, b, c in map(tris, members):
            t = np.cross(b, c) @ a / 6
            vol += t.sum()
            moment += (t[:, None] * (a + b + c)).sum(axis=0) / 4
        return vol, (moment / vol if vol else moment)

    # Is point q inside the closed piece? Add up how much of the view from q
    # each triangle covers (its solid angle). Seen from inside, the triangles
    # cover the full sphere and the result is 1. From outside, they cancel to 0.
    def winding_number(q, members):
        w = 0.0
        for a, b, c in map(tris, members):
            a, b, c = a - q, b - q, c - q
            la, lb, lc = (
                np.linalg.norm(a),
                np.linalg.norm(b, axis=1),
                np.linalg.norm(c, axis=1),
            )
            num = np.cross(b, c) @ a
            den = la * lb * lc + (b @ a) * lc + (c @ a) * lb + np.einsum("ij,ij->i", b, c) * la
            w += 2 * np.arctan2(num, den).sum()
        return w / (4 * np.pi)

    # Step 2: decide which way each piece faces. A piece is closed when every
    # one of its edges is shared by exactly two faces.
    closed = [
        all(len(edge_faces[frozenset(e)]) == 2 for fi in m for e in edges(faces[fi]))
        for m in pieces
    ]
    volumes = [volume_and_centroid(m) if c else (0.0, None) for m, c in zip(pieces, closed)]
    room = max(
        (i for i, c in enumerate(closed) if c),
        key=lambda i: abs(volumes[i][0]),
        default=None,
    )
    # Without a closed room, use the average of all vertices as the centre
    # (that's the origin, since we moved it there above).
    center = np.zeros(3)
    if room is not None:
        if volumes[room][0] < 0:
            flip(pieces[room])
        # Flipping the room doesn't move its centroid, so this is still valid.
        center = volumes[room][1]

    for i, members in enumerate(pieces):
        if i == room:
            continue
        if closed[i]:
            # An obstacle should end up with a negative volume (normals point
            # into it), anything outside the room with a positive one.
            # We check every vertex and keep the highest result. A vertex
            # touching a wall or the floor can give any result, but an obstacle
            # inside the room always has at least one vertex clear of them.
            # (The average of the vertices isn't enough: for a U-shaped
            # obstacle it can lie outside a U-shaped room.)
            idx = np.unique([v for fi in members for v in faces[fi].vertex_indices]) - 1
            inside = room is not None and max(winding_number(q, pieces[room]) for q in P[idx]) > 0.5
            if (volumes[i][0] > 0) == inside:
                flip(members)
        else:
            # shortcut: a piece with holes has no inside, so its faces vote with
            # their area for "my normal points toward the room centre". This can
            # get it wrong for a piece that wraps around the centre, like a
            # U-shaped shell with an open top; casting rays from each face to see
            # which side faces inside would fix that.
            toward = 0.0
            for fi in members:
                a, b, c = tris(fi)
                area = np.cross(b - a, c - a).sum(axis=0) / 2
                toward += area @ (center - P[np.asarray(faces[fi].vertex_indices) - 1].mean(axis=0))
            if toward > 0:
                flip(members)

    diag = {
        "pieces": len(pieces),
        "closed_pieces": sum(closed),
        "flipped_faces": sum(flipped),
    }
    if logger:
        logger.info("[ORIENT] %s", diag)
    return diag


def flip_all_faces_if_majority_inward(
    faces,
    unique_vertices,
    room_center,
    logger=None,
) -> bool:
    """Flip every face when the majority currently point inward.

    For each face, the Newell normal is compared against the vector from the
    face centroid to ``room_center``. An outward normal points away from the
    room, so that dot product is negative -> 'outward'; otherwise 'inward'.
    If inward strictly outnumbers outward, all faces are reversed.

    A tie (``inward == outward``) is deliberately resolved as *keep*: with no
    majority signal there is no reason to prefer a global flip, and keeping
    the input orientation is the least-surprising, idempotent choice. Returns
    ``True`` iff the faces were flipped.
    """
    inward = 0
    outward = 0

    for f in faces:
        vids = f.vertex_indices
        pts = [unique_vertices[i - 1] for i in vids]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        cz = sum(p[2] for p in pts) / len(pts)

        nx, ny, nz = newell_normal_from_points(vids, unique_vertices)
        to_center = (room_center[0] - cx, room_center[1] - cy, room_center[2] - cz)
        dotp = nx * to_center[0] + ny * to_center[1] + nz * to_center[2]

        if dotp < 0:
            outward += 1
        else:
            inward += 1

    if inward > outward:
        for f in faces:
            f.vertex_indices.reverse()
        if logger:
            logger.info(
                "[ORIENT] flipped ALL faces (majority inward: inward=%d outward=%d)",
                inward,
                outward,
            )
        return True

    if logger:
        logger.info(
            "[ORIENT] kept orientation (majority outward/tie: inward=%d outward=%d)",
            inward,
            outward,
        )
    return False


__all__ = ["room_center_from_mesh", "flip_all_faces_if_majority_inward"]


def get_or_create_vertex(
    points: list[tuple[float, float, float]], p: tuple[float, float, float], tol: float = 1e-9
) -> int:
    for i, q in enumerate(points, start=1):
        if distance(p, q) <= tol:
            return i
    points.append(p)
    return len(points)


def compact_vertices_and_remove_unused(
    faces: List["Face"],
    points: List[Tuple[float, float, float]],
) -> Tuple[List["Face"], List[Tuple[float, float, float]], bool, Dict[str, Any]]:
    # Mesh-only: require `Face` objects from `app.geometry.ir` with
    # a `vertex_indices` attribute. Legacy `FaceRecord` is no longer
    # accepted here — callers must be migrated to the Mesh IR.
    def _face_vids(face) -> List[int]:
        if not hasattr(face, "vertex_indices"):
            raise TypeError(
                "compact_vertices_and_remove_unused: legacy FaceRecord not accepted; "
                "migrate callers to use app.geometry.ir.Face with `vertex_indices`."
            )
        return list(face.vertex_indices)

    used = sorted({vid for f in faces for vid in _face_vids(f)})
    mapping = {old_vid: new_vid for new_vid, old_vid in enumerate(used, start=1)}
    new_points = [points[old_vid - 1] for old_vid in used]
    changed = len(used) != len(points) or any(old_vid != mapping[old_vid] for old_vid in used)
    new_faces: List["Face"] = []
    for i, f in enumerate(faces):
        vids = [mapping[vid] for vid in _face_vids(f)]
        # Always construct Mesh `Face` instances.
        new_faces.append(
            Face(
                vertex_indices=vids,
                group=getattr(f, "group", "default"),
                material=getattr(f, "material", None),
            )
        )

    diag = {
        "status": "ok",
        "vertices_before": len(points),
        "vertices_after": len(new_points),
        "removed_unused_vertices": len(points) - len(new_points),
    }
    return new_faces, new_points, changed, diag


def _find_face_by_fid(faces: List["Face"], fid: int) -> "Any | None":
    for i, face in enumerate(faces):
        if getattr(face, "fid", None) == fid:
            return face
    return None


def _endpoint_vids_from_edge_t(edge, t, *, t_eps=1e-9):
    u, v = edge

    if abs(t) <= t_eps:
        return u, v
    if abs(t - 1.0) <= t_eps:
        return v, u
    return None, None


# pure-math helpers `polygon_centroid`, `compute_face_unit_normal`,
# and `offset_point_along_vector` are provided by
# app.geometry.geometry_math.geometry_math


def move_touching_endpoint_off_face(
    faces,
    points,
    plc_report,
    *,
    offset_m=0.01,
    logger=None,
    t_eps=1e-9,
):
    diag = {
        "status": "noop",
        "touched_vid": None,
        "other_vid": None,
        "facet_fid": plc_report.get("facet_fid"),
        "old_point": None,
        "new_point": None,
        "normal": None,
        "offset_m": offset_m,
        "len_plus": None,
        "len_minus": None,
        "chosen_direction": None,
    }

    if plc_report.get("hit_type") != "endpoint_face_interior_touch":
        diag["status"] = "wrong_hit_type"
        return points, False, diag

    touched_vid, other_vid = _endpoint_vids_from_edge_t(
        plc_report["edge"],
        plc_report["t_param"],
        t_eps=t_eps,
    )

    if touched_vid is None or other_vid is None:
        diag["status"] = "not_an_endpoint_hit"
        return points, False, diag

    diag["touched_vid"] = touched_vid
    diag["other_vid"] = other_vid

    facet_fid = plc_report["facet_fid"]
    touched_face = _find_face_by_fid(faces, facet_fid)

    if touched_face is None:
        diag["status"] = "missing_facet_face"
        return points, False, diag

    # Mesh-only: require `vertex_indices` on the touched face.
    if not hasattr(touched_face, "vertex_indices"):
        raise TypeError(
            "move_touching_endpoint_off_face: legacy FaceRecord not accepted; "
            "migrate callers to use app.geometry.ir.Face with `vertex_indices`."
        )
    tf_verts = list(touched_face.vertex_indices)
    normal = compute_face_unit_normal(tf_verts, points)
    if normal is None:
        diag["status"] = "invalid_face_normal"
        return points, False, diag

    old_point = points[touched_vid - 1]
    other_point = points[other_vid - 1]

    p_plus = offset_point_along_vector(old_point, normal, +offset_m)
    p_minus = offset_point_along_vector(old_point, normal, -offset_m)

    len_plus = distance(p_plus, other_point)
    len_minus = distance(p_minus, other_point)

    diag["old_point"] = old_point
    diag["normal"] = normal
    diag["len_plus"] = len_plus
    diag["len_minus"] = len_minus

    if len_plus < len_minus:
        new_point = p_plus
        chosen = "+normal"
    else:
        new_point = p_minus
        chosen = "-normal"

    points[touched_vid - 1] = new_point

    diag["status"] = "ok"
    diag["new_point"] = new_point
    diag["chosen_direction"] = chosen

    if logger:
        logger.info(
            "[PLC OFFSET] moved vid=%d away from facet_fid=%d by %.6f m "
            "using shorter-edge rule other_vid=%d "
            "len(+n)=%.6f len(-n)=%.6f chosen=%s "
            "old=(%.6f,%.6f,%.6f) new=(%.6f,%.6f,%.6f) normal=(%.6f,%.6f,%.6f)",
            diag["touched_vid"],
            facet_fid,
            offset_m,
            diag["other_vid"],
            diag["len_plus"],
            diag["len_minus"],
            diag["chosen_direction"],
            diag["old_point"][0],
            diag["old_point"][1],
            diag["old_point"][2],
            diag["new_point"][0],
            diag["new_point"][1],
            diag["new_point"][2],
            diag["normal"][0],
            diag["normal"][1],
            diag["normal"][2],
        )

    return points, True, diag


__all__ = [
    "room_center_from_mesh",
    "polygon_centroid",
    "get_or_create_vertex",
    "compact_vertices_and_remove_unused",
]
