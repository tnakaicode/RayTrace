#!/usr/bin/env python
# coding: utf-8


import logging
# Some packages used by pvtrace are a little noisy
logging.getLogger('trimesh').disabled = True
logging.getLogger('shapely.geos').disabled = True
import numpy as np
from pvtrace import Node, Sphere, Material, Scene, MeshcatRenderer, Ray, photon_tracer
from tutorial_support import interact_ray


# # Interactive ray tracing example
#
# Let's make a scene and use ipython widgets to move the starting point of rays.
#
# All scenes must have a **world node** which contains all other objects.


world = Node(
    name="world (air)",
    geometry=Sphere(
        radius=10.0,
        material=Material(refractive_index=1.0),
    )
)
sphere = Node(
    name="sphere (glass)",
    geometry=Sphere(
        radius=1.0,
        material=Material(refractive_index=1.5),
    ),
    parent=world
)
scene = Scene(world)


# The scene is a large world sphere containing air with a smaller glass sphere inside and at the centre.
#
# ---
#
# **Units**
#
# *pvtrace* does *not* assume units, they could be metres, kilometres or lightyears.
#
# ---

# Use the MeshCat renderer to view the scene directly in the Jupyter notebook. The simplest scene just contains a Ray.


renderer = MeshcatRenderer(wireframe=True)
renderer.render(scene)
renderer.vis.jupyter_cell()


_ = interact_ray(scene, renderer)


# The `interact_ray` function makes a Ray,


ray = Ray(
    position=(0.0, 0.0, 0.0),
    direction=(0.0, 0.0, 1.0),
    wavelength=555.0
)


# and every time the one of the slides changes and retraces the scene.
#
# Rays are a simple data structure containing `position`, `direction` and `wavelength`.

# ## Getting trace data
#
# Use the `follow` function, from the `photon_tracer` module, to get a list of interaction points and event types that a ray made with the scene.


np.random.seed(0)
steps = photon_tracer.follow(scene, ray)


r, e = steps[0]
f"{e.name}: {r}"


# At the first step the ray is generated with the values we specify.


r, e = steps[1]
f"{e.name}: {r}"


# The ray hits the small glass sphere and is transmitted.


r, e = steps[2]
f"{e.name}: {r}"


# Finally the ray exits the scene by hitting the world sphere.

# Full list of events is described by the enum,
#
# ```python
# class Event(Enum):
#     GENERATE = 0
#     REFLECT = 1
#     TRANSMIT = 2
#     ABSORB = 3
#     SCATTER = 4
#     EMIT = 5
#     EXIT = 6
#     KILL = 7
# ```

# This gives a hint of the other capabilities of pvtrace which will be covered in the next tutorials.
