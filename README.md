## Examples

Each example in the `Examples` directory contains the following files:

* `.hlang` is the I❤️MESH source code
* `.pdf` is the typeset generated LaTeX.
* `iheartmesh.h` is the generated C++ code.
* `main.cpp` is the driver application.

## Compiling

You can compile all the examples at once by running the following command in the current folder:

```mkdir build && cd build && cmake .. && make```

You can compile a specific example by entering the corresponding `Examples/*` folder:

```cd Examples/*```

and running the same commands:

```mkdir build && cd build && cmake .. && make```

## The `resource` folder

* A video of the mass-spring example `bunny_to_the_ground.mp4`
* The `include` and `src` folders contain the C++ dependencies for the C++ code generated by I❤️MESH.
* The `models` folder contains data used by the examples.
* The `iheartmesh` folder contains the source code for the I❤️MESH compiler. Neighbor files are in the `iheartmesh/iheartla/mesh` folder.
