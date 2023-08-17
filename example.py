import torch
import mano
from mano.utils import Mesh


right_path = './models/mano/MANO_RIGHT.pkl'
left_path = './models/mano/MANO_LEFT.pkl'
n_comps = 45
batch_size = 10


# generate right hand
rh_model = mano.load(model_path=right_path,
                     is_right=True,
                     num_pca_comps=n_comps,
                     batch_size=batch_size,
                     flat_hand_mean=False)

rh_betas = torch.rand(batch_size, 10)*.1
rh_pose = torch.rand(batch_size, n_comps)*.1
rh_global_orient = torch.rand(batch_size, 3)
rh_transl        = torch.rand(batch_size, 3)*.2

rh_output = rh_model(betas=rh_betas,
                     global_orient=rh_global_orient,
                     hand_pose=rh_pose,
                     transl=rh_transl,
                     return_verts=True,
                     return_tips=True)

rh_meshes = rh_model.hand_meshes(rh_output)
rj_meshes = rh_model.joint_meshes(rh_output)


# generate left hand
lh_model = mano.load(model_path=left_path,
                     is_right=False,
                     num_pca_comps=n_comps,
                     batch_size=batch_size,
                     flat_hand_mean=False)

lh_betas = torch.rand(batch_size, 10)*.1
lh_pose = torch.rand(batch_size, n_comps)*.1
lh_global_orient = torch.rand(batch_size, 3)
lh_transl        = torch.rand(batch_size, 3)*.2

lh_output = lh_model(betas=lh_betas,
                     global_orient=lh_global_orient,
                     hand_pose=lh_pose,
                     transl=lh_transl,
                     return_verts=True,
                     return_tips=True)

lh_meshes = lh_model.hand_meshes(lh_output)
lj_meshes = lh_model.joint_meshes(lh_output)


# visualize hand mesh only
rh_meshes[0].show()
lh_meshes[0].show()

# visualize joints mesh only
rj_meshes[0].show()
lj_meshes[0].show()

# visualize hand and joint meshes
hj_meshes = Mesh.concatenate_meshes([rh_meshes[0], rj_meshes[0], lh_meshes[0], lj_meshes[0]])
hj_meshes.show()
