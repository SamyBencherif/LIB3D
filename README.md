# **LIB3D** is a 3d graphics library intended to work in [Pythonista](http://omz-software.com/pythonista/) on iOS

Supports

* isometric quad rendering
    * textures are selected from Pythonista's builtins
    * coordinates are specified in 3d using (3,4) matrices
* directional shading
    * a single directional light source is currently applied to all scenes
* user-interface controls
    * buttons (with text and/or image)
    * toggles (with text and/or image)
    * selection group (with text and/or image)
    * sliders
    * joysticks
* efficient voxel-quad construction
    * 2 cubes placed next to each other result in 10 quads instead of 12
    * 4 cubes packed have 16 quads instead of 24
    * worldBuilder.py can be used to interactively build voxel worlds

## Notes

I am building this module so I can write the same 3d programs from my phone and my computer.  

* To be clear, the renderer is isometric, but it could be theoretically made perspective by adjusting the default fragment shader to include perspective correction and changing the projection matrix.

* The other constraint, voxels, has nothing to do with the renderer. This is a result of the existing scene management functions. It would be possible to incorporate non-voxel based geometry by adding supporting functions.

* Finally, we have another rendering constaint. Currently we only draw quads because LIB3D uses image_quad from Pythonista's scene module. Incidentally there is also a triangle_strip function.

## How to Run

This project is intended for iPhone, but it can also be run on PC using a port I am also developing:
    [Compista](about:blank)