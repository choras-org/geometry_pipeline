=======
History
=======

Unreleased
----------

Bug Fixes
~~~~~~~~~

- Write a single volume instead of failing when the cavity detector fails or times out (``VOLUME_DETECTOR_TIMEOUT``) (#6)

0.1.2 (2026-11-20)
------------------

Bug Fixes
~~~~~~~~~

- Orient faces outward per connected piece (#4)
- Fix materials assigned to the wrong surfaces in the ``.geo`` export (#5)

Deprecations
~~~~~~~~~~~~

- Deprecate ``FlipFacesIfMajorityInwardRepair``, to be removed in 0.2.0 (#4)

0.1.1 (2026-11-19)
------------------

Bug Fixes
~~~~~~~~~

- Write physical surfaces before physical volumes in the ``.geo`` export, so each surface keeps the physical tag of its surface entity. Simulations with the DG method failed for geometries exported by the pipeline

0.1.0 (2026-11-18)
------------------

Initial version: import, validation, repair and export of room geometry for CHORAS, with the native CGAL volume detector built into the wheel
