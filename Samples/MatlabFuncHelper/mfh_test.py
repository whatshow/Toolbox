import numpy as np
from MatlabFuncHelper import MatlabFuncHelper

mfh = MatlabFuncHelper();

# seq
rtn_seq = mfh.seq(3,4);
assert(rtn_seq.shape[-1] == 1);
rtn_seq = mfh.seq(3,4, order="F");
assert(rtn_seq.shape[-1] == 2);

# ones
mfh.batch_size = None;
rtn_ones = mfh.ones(4, order="F");
assert(rtn_ones.shape == (4,4));
rtn_ones = mfh.ones(4, order="C");
assert(rtn_ones.shape == (4,));
mfh.batch_size = 18;
rtn_ones = mfh.ones(4, order="F");
assert(rtn_ones.shape == (18,4,4));
rtn_ones = mfh.ones(4, order="C");
assert(rtn_ones.shape == (18,4));

# eye
mfh.batch_size = None;
rtn_eye = mfh.eye(4);
assert(rtn_eye.shape == (4,4));
rtn_eye = mfh.eye(4, 8);
assert(rtn_eye.shape == (4,8));
mfh.batch_size = 18;
rtn_eye = mfh.eye(4);
assert(rtn_eye.shape == (18,4,4));
rtn_eye = mfh.eye(4, 8);
assert(rtn_eye.shape == (18,4,8));

# rand
mfh.batch_size = None;
rtn_rand = mfh.rand(4);
assert(rtn_rand.shape == (4,));
mfh.batch_size = 18;
rtn_rand = mfh.rand(4);
assert(rtn_rand.shape == (18,4));

# repmat
# repmat - rns is short than input matrix (if use batch size, extra dimension will be added)
mfh.batch_size = None;
rtn_repmat = mfh.repmat1([[1, 2]], 2, order="F");
assert(np.sum(rtn_repmat - np.tile([1, 2], (2,2)), axis=None) == 0);
assert(rtn_repmat.ndim == 2);
rtn_repmat = mfh.repmat1([[1, 2]], 2, order="C");
assert(np.sum(rtn_repmat - np.tile([1, 2], (1,2)), axis=None) == 0);
assert(rtn_repmat.ndim == 2);
rtn_repmat = mfh.repmatN([[1, 2]], 2, order="F");
assert(np.sum(rtn_repmat - np.tile([1, 2], (2,2)), axis=None) == 0);
assert(rtn_repmat.ndim == 2);
rtn_repmat = mfh.repmatN([[1, 2]], 2, order="C");
assert(np.sum(rtn_repmat - np.tile([1, 2], (1,2)), axis=None) == 0);
assert(rtn_repmat.ndim == 2);
mfh.batch_size = 20;
rtn_repmat = mfh.repmat1([[1, 2]], 2, order="F");
assert(np.sum(rtn_repmat - np.tile([1, 2], (1,2,2)), axis=None) == 0);
assert(rtn_repmat.ndim == 3);
rtn_repmat = mfh.repmat1([[1, 2]], 2, order="C");
assert(np.sum(rtn_repmat - np.tile([1, 2], (1,1,2)), axis=None) == 0);
assert(rtn_repmat.ndim == 3);
rtn_repmat = mfh.repmatN([[1, 2]], 2, order="F");
assert(np.sum(rtn_repmat - np.tile([1, 2], (20,2,2)), axis=None) == 0);
assert(rtn_repmat.ndim == 3);
rtn_repmat = mfh.repmatN([[1, 2]], 2, order="C");
assert(np.sum(rtn_repmat - np.tile([1, 2], (20,1,2)), axis=None) == 0);
assert(rtn_repmat.ndim == 3);
# repmat - rns is longer than input matrix (if use batch size, the actual batch_size is ignored)
mfh.batch_size = None;
rtn_repmat = mfh.repmat1([[1, 2]], 2, 3, 4, order="F");
assert(np.sum(rtn_repmat - np.tile([1, 2], (2,3,4)), axis=None) == 0);
assert(rtn_repmat.ndim == 3);
rtn_repmat = mfh.repmat1([[1, 2]], 2, 3, 4, order="C");
assert(np.sum(rtn_repmat - np.tile([1, 2], (2,3,4)), axis=None) == 0);
assert(rtn_repmat.ndim == 3);
rtn_repmat = mfh.repmatN([[1, 2]], 2, 3, 4, order="F");
assert(np.sum(rtn_repmat - np.tile([1, 2], (2,3,4)), axis=None) == 0);
assert(rtn_repmat.ndim == 3);
rtn_repmat = mfh.repmatN([[1, 2]], 2, 3, 4, order="C");
assert(np.sum(rtn_repmat - np.tile([1, 2], (2,3,4)), axis=None) == 0);
assert(rtn_repmat.ndim == 3);
mfh.batch_size = 20;
rtn_repmat = mfh.repmat1([[1, 2]], 2, 3, 4, order="F");
assert(np.sum(rtn_repmat - np.tile([1, 2], (2,3,4)), axis=None) == 0);
assert(rtn_repmat.ndim == 3);
rtn_repmat = mfh.repmat1([[1, 2]], 2, 3, 4, order="C");
assert(np.sum(rtn_repmat - np.tile([1, 2], (2,3,4)), axis=None) == 0);
assert(rtn_repmat.ndim == 3);
rtn_repmat = mfh.repmatN([[1, 2]], 2, 3, 4, order="F");
assert(np.sum(rtn_repmat - np.tile([1, 2], (2,3,4)), axis=None) == 0);
assert(rtn_repmat.ndim == 3);
rtn_repmat = mfh.repmatN([[1, 2]], 2, 3, 4, order="C");
assert(np.sum(rtn_repmat - np.tile([1, 2], (2,3,4)), axis=None) == 0);
assert(rtn_repmat.ndim == 3);

# kron
mfh.batch_size = 2;
rtn_kron = mfh.kron([[1, 2], [1, 3]], [[[2],[3]], [[2],[3]]]);
assert(np.sum(rtn_kron, axis=None) - 5*2-5*5 == 0);
