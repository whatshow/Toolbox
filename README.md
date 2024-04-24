# Toolbox
[![PyPi](https://img.shields.io/badge/PyPi-1.0.1-blue)](https://pypi.org/project/whatshow-phy-mod-otfs/) [![MathWorks](https://img.shields.io/badge/MathWorks-1.0.1-red)](https://mathworks.com/matlabcentral/fileexchange/161136-whatshow_phy_mod_otfs)


This repositories offers a toolbox across platforms.
## How to install
* Install through `Matlab Add-Ons`
    * Install through Matlab `Get Add-Ons`: search `whatshow_utils` and install it.
    * Install through `.mltbx`: Go to ***Releases*** to download the file `*.mltbx` in the latest release to install.
* Install through `pip`
    ```sh
    pip install whatshow-utils
    ```
    * **import this module**
        ```
        from whatshow_utils import OTFS
        ```
        
### How to use
* MatlabFuncHelper:
    * checkers
        * isvector()
        * ismatrix()
    * generators
        * zeros()
        * ones()
        * eye()
        * seq(): generate a sequence
        * rand():
        * randn():
        * dftmtx(): generate the discrete transform matrix
    * transformers
        * squeeze(): remove redundant dimension except for the batch_size
        * reshape():
        * repmat1(): repeat the matrix in the given dimension (the batch dim is repeated as 1)
        * repmatN(): repeat the matrix in the given dimension (the batch dim is repeated as batch_size)
        * diag(): generate a matrix based on its diag or get a diagonal matrix from its vector<br>
            `@mat`: a vector as [(batch_size), n, 1] or [(batch_size), 1, n]; if n == 1, it will be taken as [(batch_size), n, 1]. Or a square matrix [(batch_size), n, n]
        * circshift(): circular shift (1st index except for the batch size)<br>
            `@mat`: [(batch_size), dim1, dim2, ...]<br>
            `@step`: a scalar or an 1D array (only for batch)
    * Maths
        * max(): return the maximum value of a matrix or the maximu value of two matrices (for complex value, we compare the magnitude)
        * kron(): Kronecker product