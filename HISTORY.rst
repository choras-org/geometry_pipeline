=======
History
=======

Unreleased
----------

Bug Fixes
~~~~~~~~~

- Write physical surfaces before physical volumes in the ``.geo`` export, so each surface keeps the physical tag of its surface entity. Simulations with the DG method failed for geometries exported by the pipeline

0.1.0 (2026-11-18)
------------------

Initial version: import, validation, repair and export of room geometry for CHORAS, with the native CGAL volume detector built into the wheel
