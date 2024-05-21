import numpy as np
import torch as pt
import tensorflow as tf
from MatlabFuncHelper import MatlabFuncHelper


bs = 5;

# test no batch data, 1 batch, n batch
mfh = MatlabFuncHelper();
mfh_b1 = MatlabFuncHelper();
mfh_b1.setBatchSize(1);
mfh_bn = MatlabFuncHelper();
mfh_bn.setBatchSize(bs);

'''
isvector & ismatrix
'''
# build data - scalars
sca_py = 1;
sca_np = np.array(1);
sca_pt = pt.tensor(1);
sca_tf = tf.Variable(1);
sca_py_b1 = np.tile(1, 1).tolist();
sca_np_b1 = np.tile(1, 1);
sca_pt_b1 = pt.tile(sca_pt, [1]);
sca_tf_b1 = tf.tile(tf.expand_dims(sca_tf, -1), [1]);
sca_py_bn = np.tile(1, bs).tolist();
sca_np_bn = np.tile(1, bs);
sca_pt_bn = pt.tile(sca_pt, [bs]);
sca_tf_bn = tf.tile(tf.expand_dims(sca_tf, -1), [bs]);

# build data
# build data - vectors
vec_py = [ [1], [1,2,3], [[1,2,3]], [[1],[2],[3]] ];
vec_np = [ np.array([1]), np.array([1,2,3]), np.array([[1,2,3]]), np.array( [[1],[2],[3]]) ];
vec_pt = [ pt.tensor([1]), pt.tensor([1,2,3]), pt.tensor([[1,2,3]]), pt.tensor( [[1],[2],[3]])];
vec_tf = [ tf.Variable([1]), tf.Variable([1,2,3]), tf.Variable([[1,2,3]]), tf.Variable( [[1],[2],[3]])];
vec_py_b1 = [];
vec_np_b1 = [];
vec_pt_b1 = [];
vec_tf_b1 = [];
vec_py_bn = [];
vec_np_bn = [];
vec_pt_bn = [];
vec_tf_bn = [];
for i in range(len(vec_py)):
    tmp_vec_ndim = np.asarray(vec_py[i]).ndim;
    tmp_vec_sh_b1 = [1]*tmp_vec_ndim;
    tmp_vec_sh_bn = [1]*tmp_vec_ndim;
    tmp_vec_sh_b1.insert(0, 1);
    tmp_vec_sh_bn.insert(0, bs);
    vec_py_b1.append(np.tile(vec_py[i], tmp_vec_sh_b1).tolist());
    vec_py_bn.append(np.tile(vec_py[i], tmp_vec_sh_bn).tolist());
    vec_np_b1.append(np.tile(vec_np[i], tmp_vec_sh_b1));
    vec_np_bn.append(np.tile(vec_np[i], tmp_vec_sh_bn));
    vec_pt_b1.append(pt.tile(vec_pt[i], tmp_vec_sh_b1));
    vec_pt_bn.append(pt.tile(vec_pt[i], tmp_vec_sh_bn));
    
    vec_tf_b1.append(tf.tile(tf.expand_dims(vec_tf[i], 0), tmp_vec_sh_b1));
    vec_tf_bn.append(tf.tile(tf.expand_dims(vec_tf[i], 0), tmp_vec_sh_bn));

# check
# check - scalars
# check - scalars - isvector
assert(not mfh.isvector(sca_py));
assert(not mfh.isvector(sca_np));
assert(not mfh.isvector(sca_pt));
assert(not mfh.isvector(sca_tf));
assert(not mfh_b1.isvector(sca_py_b1));
assert(not mfh_b1.isvector(sca_np_b1));
assert(not mfh_b1.isvector(sca_pt_b1));
assert(not mfh_b1.isvector(sca_tf_b1));
assert(not mfh_bn.isvector(sca_py_bn));
assert(not mfh_bn.isvector(sca_np_bn));
assert(not mfh_bn.isvector(sca_pt_bn));
assert(not mfh_bn.isvector(sca_tf_bn));
# check - scalars - ismatrix
assert(not mfh.ismatrix(sca_py));
assert(not mfh.ismatrix(sca_np));
assert(not mfh.ismatrix(sca_pt));
assert(not mfh.ismatrix(sca_tf));
assert(not mfh_b1.ismatrix(sca_py_b1));
assert(not mfh_b1.ismatrix(sca_np_b1));
assert(not mfh_b1.ismatrix(sca_pt_b1));
assert(not mfh_b1.ismatrix(sca_tf_b1));
assert(not mfh_bn.ismatrix(sca_py_bn));
assert(not mfh_bn.ismatrix(sca_np_bn));
assert(not mfh_bn.ismatrix(sca_pt_bn));
assert(not mfh_bn.ismatrix(sca_tf_bn));
# check - vectors
for i in range(len(vec_py)):
    # isvector
    assert(mfh.isvector(vec_py[i]));
    assert(mfh.isvector(vec_np[i]));
    assert(mfh.isvector(vec_pt[i]));
    assert(mfh.isvector(vec_tf[i]));
    assert(mfh_b1.isvector(vec_py_b1[i]));
    assert(mfh_b1.isvector(vec_np_b1[i]));
    assert(mfh_b1.isvector(vec_pt_b1[i]));
    assert(mfh_b1.isvector(vec_tf_b1[i]));
    assert(mfh_bn.isvector(vec_py_bn[i]));
    assert(mfh_bn.isvector(vec_np_bn[i]));
    assert(mfh_bn.isvector(vec_pt_bn[i]));
    assert(mfh_bn.isvector(vec_tf_bn[i]));
    # ismatrix
    assert(not mfh.ismatrix(vec_py[i]));
    assert(not mfh.ismatrix(vec_np[i]));
    assert(not mfh.ismatrix(vec_pt[i]));
    assert(not mfh.ismatrix(vec_tf[i]));
    assert(not mfh_b1.ismatrix(vec_py_b1[i]));
    assert(not mfh_b1.ismatrix(vec_np_b1[i]));
    assert(not mfh_b1.ismatrix(vec_pt_b1[i]));
    assert(not mfh_b1.ismatrix(vec_tf_b1[i]));
    assert(not mfh_bn.ismatrix(vec_py_bn[i]));
    assert(not mfh_bn.ismatrix(vec_np_bn[i]));
    assert(not mfh_bn.ismatrix(vec_pt_bn[i]));
    assert(not mfh_bn.ismatrix(vec_tf_bn[i]));
# check - matrix


'''
isnan
'''
    