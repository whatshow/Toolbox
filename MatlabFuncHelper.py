import numpy as np
class UtilMatlabFunc(object):
    BATCH_SIZE_NO = None;
    batch_size = 0;
    
    ###########################################################################
    # checkers
    '''
    check input is a vector like [(batch_size), n],  [(batch_size), n, 1] or [(batch_size), 1, n] 
    '''
    def isvector(self, mat):
        mat = np.asarray(mat);
        if self.batch_size is self.BATCH_SIZE_NO:
            return mat.ndim == 1 or mat.ndim == 2 and (mat.shape[-2] == 1 or mat.shape[-1] == 1);
        else:
            if mat.shape[0] != self.batch_size:
                raise Exception("The input does not has the required batch size.");
            else:
                return mat.ndim == 2 or mat.ndim == 3 and (mat.shape[-2] == 1 or mat.shape[-1] == 1);
            
    '''
    check input is a matrix like [(batch_size), n. m]
    '''
    def ismatrix(self, mat):
        mat = np.asarray(mat);
        if self.batch_size is self.BATCH_SIZE_NO:
            return mat.ndim == 2 and mat.shape[-2] > 1 and mat.shape[-1] > 1;
        else:
            if mat.shape[0] != self.batch_size:
                raise Exception("The input does not has the required batch size.");
            else:
                return mat.ndim == 3 and mat.shape[-2] > 1 and mat.shape[-1] > 1;
    
    ###########################################################################
    # matrix operation
    '''
    generate a matrix of all zeros
    @order: 'C': this function only create given dimensions; 'F': create the dimensions as matlab (2D at least)
    '''
    def zeros(self, nrow, *args, order='C'):
        out = None;
        if order == 'F':
            ncol = args[0] if len(args) >= 1 else nrow;
            out = np.zeros((nrow, ncol)) if self.batch_size == self.BATCH_SIZE_NO else np.zeros((self.batch_size, nrow, ncol));
        elif order == 'C':
            zeros_shape = list(args);
            zeros_shape.insert(0, nrow);
            if self.batch_size != self.BATCH_SIZE_NO:
                zeros_shape.insert(0, self.batch_size);
            out = np.zeros(zeros_shape);
        return out;
    
    '''
    generate a matrix full of ones
    @order: 'C': this function only create given dimensions; 'F': create the dimensions as matlab (2D at least)
    '''
    def ones(self, nrow, *args, order='C'):
        shape = [];
        # format the shape for dimensions
        if order == 'F':
            ncol = nrow;
            if len(args) >= 1:
                ncol = args[0];
            shape.append(nrow);
            shape.append(ncol);
        elif order == 'C':
            shape = list(args);
            shape.insert(0, nrow);
        else:
            raise Exception("The order is illegal.");
        # format the shape for batch_size
        if self.batch_size != self.BATCH_SIZE_NO:
            shape.insert(0, self.batch_size);
        return np.ones(shape);
    
    '''
    return an identity matrix
    @size: a tuple of the shape
    '''
    def eye(self, nrow, *args):
        if len(args) == 0:
            shape = [nrow];
            shape.append(nrow);
        else:
            shape = 
        
        out = np.eye(size);
        if self.batch_size is not self.BATCH_SIZE_NO:
            out = np.tile(out,(self.batch_size, 1, 1));
        return out;