<p align="center">
  <img src="/code/files/logo.png" alt="logo" width="150"/>
</p>
<h1 align="center">Photoluminescence Modeling Program for Potential Fluctuations</h1>

Repository for the source code of a Python-based photoluminescence modeling program for the study of potential fluctuations. Existing models for band gap fluctuations<sup>[1,2]</sup> and electrostatic fluctuations<sup>[3]</sup> are implemented, along with a unified potential fluctuations model that merges both fluctuations<sup>[4]</sup>. A correction for interference is also possible using the interference function.<sup>[5]</sup>

## Installation
The code has been tested on [Python 3.8.8](https://www.python.org/downloads/release/python-388/ "https://www.python.org/downloads/release/python-388/") (64-bit). Run this line on the command line to install the necessary modules to run the program:
```
pip install scipy numpy pandas matplotlib Pillow
```
To run the program, simply run
```
python InteractivePLFittingGUI.py
```
from the [code](./code) directory. Alternatively, you can run the executable file provided in the same directory, which does not require a local Python installation (only for Windows users). In this case, Windows might flag the file as risky given that it is not created by a registered publisher. The file is safe! You can also always make your own executable (see below).

The program has been tested with the following module versions:
+ scipy: 1.6.2
+ numpy: 1.20.2
+ pandas: 1.2.3
+ matplotlib: 3.3.4
+ Pillow: 8.1.2

## Making a new executable
The executable was created with [PyInstaller](https://pypi.org/project/pyinstaller/ "https://pypi.org/project/pyinstaller/"), which bundles all the necessary Python scripts, modules and data files into one executable.
If local changes to the source code are done, you can compile a new executable with
```
pyinstaller InteractivePLFittingGUI.spec
```
This has been tested for the module versions mentioned above and pyinstaller v4.2, which can also be installed via `pip`. This executable provides no error logging in case it has been incorrectly generated. If such is the case, you can change the `console` parameter to `True` in the `exe` variable defined in `InteractivePLFittingGUI.spec`. This will log the output of the executable to the console from which it is run.

## Program manual
The [manual](./program_manual.pdf) introduces the working of the program and its functionalities.

## References
[1] U. Rau and J. H. Werner, "Radiative efficiency limits of solar cells with lateral band-gap fluctuations", Appl. Phys. Lett. 84, 3735-3737 (2004) https://doi.org/10.1063/1.1737071

[2] Julian Mattheis, Uwe Rau, and Jürgen H. Werner, "Light absorption and emission in semiconductors with band gap fluctuations—A study on Cu(In,Ga)Se2 thin films", Journal of Applied Physics 101, 113519 (2007) https://doi.org/10.1063/1.2721768

[3] John K. Katahara and Hugh W. Hillhouse, "Quasi-Fermi level splitting and sub-bandgap absorptivity from semiconductor photoluminescence", Journal of Applied Physics 116, 173504 (2014) https://doi.org/10.1063/1.4898346

[4] E. M. Spaans, J. de Wild, T. J. Savenije, and B. Vermang, "Unified potential fluctuations model for photoluminescence spectra at room temperature—Cu(In,Ga)Se2 thin films", Journal of Applied Physics 130, 123103 (2021) https://doi.org/10.1063/5.0056629

[5] J. K. Larsen, S.-Y. Li, J. J. S. Scragg, Y. Ren, C. Hägglund, M. D. Heinemann, S. Kretzschmar, T. Unold, and C. Platzer-Björkman, "Interference effects in photoluminescence spectra of Cu2ZnSnS4 and Cu(In,Ga)Se2 thin films", Journal of Applied Physics 118, 035307 (2015) https://doi.org/10.1063/1.4926857
