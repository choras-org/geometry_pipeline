=======
History
=======

Unreleased
----------

Bug Fixes
~~~~~~~~~

- Orient faces outward per connected piece instead of flipping all faces when most point inward. Neighbouring faces are made consistent, the room's normals point out of the room and an obstacle's into the obstacle; faces wound against their neighbours were previously kept as they were in the ``.geo`` export (ported from CHORAS backend #150)

Deprecations
~~~~~~~~~~~~

- ``FlipFacesIfMajorityInwardRepair`` is deprecated in favour of ``OrientFacesOutwardRepair`` and no longer used by the default profile. It will be removed in 0.2.0

0.1.1 (2026-11-19)
------------------

Bug Fixes
~~~~~~~~~

- Write physical surfaces before physical volumes in the ``.geo`` export, so each surface keeps the physical tag of its surface entity. Simulations with the DG method failed for geometries exported by the pipeline

0.1.0 (2026-11-18)
------------------

Initial version: import, validation, repair and export of room geometry for CHORAS, with the native CGAL volume detector built into the wheel
