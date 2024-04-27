# Toolbox
[![PyPi](https://img.shields.io/badge/PyPi-1.0.4-blue)](https://pypi.org/project/whatshow-toolbox/)


This repositories offers a toolbox across platforms.
## How to install
* Install through `Matlab Add-Ons`
    * Install through Matlab `Get Add-Ons`: search `whatshow_toolbox` and install it.
* Install through `pip`
    ```sh
    pip install whatshow-toolbox
    ```
    * **import this module**
        ```
        from whatshow_toolbox import *
        ```
        
## How to use
* MatlabFuncHelper: this class simulate all Matlab functions in python (batch is supported).
    * batch
        * setBatchSize(batch_size)
        * getBatchSize()
    * checkers
        * isvector(mat)
        * ismatrix(mat)
    * generators
        * zeros(d1, d2, ...)
        * ones(d1, d2, ...)
        * eye(d1, ...): `eye(d1)` or `eye(d1, d2)`
        * seq(): generate a sequence,  `seq(end)`, `seq(beg, end)`, `seq(beg, step, end)`
        * rand(d1, d2, ...):
        * randn(d1, d2, ...):
        * dftmtx(d1): generate the discrete transform matrix
    * transformers
        * squeeze(mat): remove redundant dimension except for the batch_size
        * reshape(mat, d1, d2, ...):
        * repmat1(mat, d1, d2, ...): repeat the matrix in the given dimension (the batch dim is repeated as 1)
        * repmatN(mat, d1, d2, ...): repeat the matrix in the given dimension (the batch dim is repeated as batch_size)
        * diag(mat): generate a matrix based on its diag or get a diagonal matrix from its vector<br>
            `@mat`: a vector as [(batch_size), n, 1] or [(batch_size), 1, n]; if n == 1, it will be taken as [(batch_size), n, 1]. Or a square matrix [(batch_size), n, n]
        * circshift(mat, step): circular shift (1st index except for the batch size)
    * Maths
        * max(): return the maximum of a matrix or the maximum of two matrices (for complex value, we compare the magnitude)
            ```c, matlab, python
            self.max(mat1, axis=-1);    // return the maximum of a matrix, axis tells which axis to look at
            self.max(mat1, 4);          // return the maximum of a matrix and the given value
            self.max(mat1, mat2);       // return the maximum of two matrices
            ```
        * kron(a, b): Kronecker product