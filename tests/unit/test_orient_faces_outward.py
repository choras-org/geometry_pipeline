"""Tests for ``orient_faces_outward`` on rooms where a global flip isn't enough.

Ported from CHORAS backend PR #150.
"""

from __future__ import annotations

import numpy as np

from geometry_pipeline.core.ir import Face
from geometry_pipeline.repairs.mesh._common import orient_faces_outward

# Floor plan of an L-shaped room. Walls 2 and 3 form the inner corner.
L_PLAN = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
# Floor plan of a small cuboid floating inside the room, like a hanging reflector.
CUBOID_PLAN = [(0.3, 0.3), (0.6, 0.3), (0.6, 0.6), (0.3, 0.6)]


def prism(plan, z0, z1, top=True):
    """Walls, floor and (optionally) ceiling of a shape with the given floor plan.

    ``plan`` goes round counter-clockwise; the returned faces (lists of corner
    points) have normals pointing out of the shape.
    """
    n = len(plan)
    faces = [
        [
            (*plan[i], z0),
            (*plan[(i + 1) % n], z0),
            (*plan[(i + 1) % n], z1),
            (*plan[i], z1),
        ]
        for i in range(n)
    ]
    faces.append([(x, y, z0) for x, y in reversed(plan)])
    if top:
        faces.append([(x, y, z1) for x, y in plan])
    return faces


def area_vector(pts):
    """Face normal scaled by the face area."""
    p = np.asarray(pts, dtype=float)
    return np.cross(p[1:-1] - p[0], p[2:] - p[0]).sum(axis=0) / 2


def wrong_faces(coord_faces, scrambled, expected):
    """Reverse the ``scrambled`` faces, run the fix, return the faces pointing the wrong way."""
    points, index = [], {}
    faces = []
    for k, f in enumerate(coord_faces):
        for p in f:
            if p not in index:
                points.append(p)
                index[p] = len(points)
        idx = [index[p] for p in f]
        faces.append(
            Face(
                vertex_indices=idx[::-1] if k in scrambled else idx, group="default", material=None
            )
        )
    orient_faces_outward(faces, points)
    result = [area_vector([points[i - 1] for i in f.vertex_indices]) for f in faces]
    return [k for k, (got, want) in enumerate(zip(result, expected)) if got @ want <= 0]


def test_l_room_with_obstacle():
    room = prism(L_PLAN, 0, 3)
    cuboid = prism(CUBOID_PLAN, 1, 1.5)
    # Room faces point out of the room, the obstacle's faces into the obstacle.
    expected = [area_vector(f) for f in room] + [-area_vector(f) for f in cuboid]
    # Turn around both inner-corner walls, the floor and one side of the cuboid.
    scrambled = {2, 3, 6, len(room) + 1}
    assert wrong_faces(room + cuboid, scrambled, expected) == []


def test_u_obstacle_in_u_room():
    # A U-shaped obstacle wrapping around the end of the gap between the room's
    # arms. The average of its vertices lies in that gap, outside the room.
    u_room = [(0, 0), (3, 0), (3, 3), (2, 3), (2, 1), (1, 1), (1, 3), (0, 3)]
    u_obstacle = [
        (0.5, 0.5),
        (2.5, 0.5),
        (2.5, 2.5),
        (2.2, 2.5),
        (2.2, 0.8),
        (0.8, 0.8),
        (0.8, 2.5),
        (0.5, 2.5),
    ]
    room = prism(u_room, 0, 3)
    obstacle = prism(u_obstacle, 1, 2)
    expected = [area_vector(f) for f in room] + [-area_vector(f) for f in obstacle]
    assert wrong_faces(room + obstacle, {0, len(room)}, expected) == []


def test_open_l_room_wound_inward():
    # A room without a ceiling, where every face points the wrong way except
    # one inner-corner wall.
    room = prism(L_PLAN, 0, 3, top=False)
    expected = [area_vector(f) for f in room]
    scrambled = set(range(len(room))) - {3}
    assert wrong_faces(room, scrambled, expected) == []
