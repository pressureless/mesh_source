# I❤️MESH: A DSL for Mesh Processing

I❤️MESH is forked from I❤️LA.

## Running

To run the desktop GUI:

    python3 -m iheartla

You can also run as a command-line compiler:

    python3 -m iheartla --help

## Installing Dependencies

I❤️MESH depends on Python 3.x (>= Python 3.9) and several modules.

Install the modules via `pip`:

    # Create a virtual environment:
    python3 -m venv .venv
    # Activate your virtual environment (shell dependent). For example:
    source .venv/bin/activate
    
    # Install the list of packages from a file:
    pip3 install -r requirements.txt
    # or directly:
    pip3 install tatsu==5.8.3 regex appdirs wxpython PyMuPDF sympy
    
    # For development, also install:
    pip3 install graphviz cppyy numpy scipy pyinstaller

(Note: There was a known bug with Python >= 3.10 and wxPython 4.2.0's PDF viewer. It was fixed in wxPython 4.2.1.)

## Output Dependencies

To use the code output for the various backends, you will need:

* LaTeX: A working tex distribution with `xelatex` and `pdfcrop`
* C++: Eigen. Compilation differs on different platforms. On macOS with Homebrew eigen: `c++ -I/usr/local/eigen3 output.cpp -o output`* Python: NumPy or Jax and SciPy
* MATLAB: MATLAB or (untested) Octave

## Unicode Fonts

`DejaVu Sans Mono` is a font with good Unicode support. Windows users should install it. You can download it [here](https://dejavu-fonts.github.io/Download.html). The I❤️LA GUI will use it if installed.
