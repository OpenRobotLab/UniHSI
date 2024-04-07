import wandb


import torch

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils
import open3d as o3d


import json

def load_label(fn):
    with open(fn, 'r') as fin:
        lines = [item.rstrip() for item in fin]
        label = np.array([int(line) for line in lines], dtype=np.int32)
        return label

def get_leaf_node(dic, results, offset):
    for i in range(len(results)):
        result = results[i]
        if 'children' in result.keys():
            get_leaf_node(dic, result['children'], offset)
        else:
            dic[result['name']+str(result['id'])] = result['id'] + offset

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def get_height_map(points: np.ndarray, HEIGHT_MAP_DIM: int=16):
    """ Load region meshes of a scenes

    Args:
        points: scene point cloud
        HEIGHT_MAP_DIM: height map dimension
    
    Return:
        Return the floor height map and axis-aligned scene bounding box
    """

    ## compute floor height map
    minx, miny = points[:, 0].min(), points[:, 1].min()
    maxx, maxy = points[:, 0].max(), points[:, 1].max()

    interval_x = (maxx-minx)/HEIGHT_MAP_DIM
    interval_y = (maxy-miny)/HEIGHT_MAP_DIM
    voxel_idx_x = (points[:, 0]-minx) // interval_x
    voxel_idx_y = (points[:, 1]-miny) // interval_y

    height_map2d = np.zeros((HEIGHT_MAP_DIM,HEIGHT_MAP_DIM))
    for i in range(HEIGHT_MAP_DIM):
        for j in range(HEIGHT_MAP_DIM):
            mask = (voxel_idx_x==i)&(voxel_idx_y==j)
            if mask.sum()==0:
                pass
            else:
                height_map2d[j,i] = points[mask, 2].max()

    x = np.linspace(minx, maxx, HEIGHT_MAP_DIM)
    y = np.linspace(miny, maxy, HEIGHT_MAP_DIM)
    xx, yy = np.meshgrid(x, y)
    # pos2d = np.concatenate([yy[..., None], xx[..., None]], axis=-1)
    pos2d = np.concatenate([xx[..., None], yy[..., None]], axis=-1)

    height_pcd = np.concatenate([pos2d.reshape(-1,2),height_map2d.reshape(-1,1)], axis=-1)

    return height_pcd

def VaryPoint(data, axis, degree):
    xyzArray = {
        'X': np.array([[1, 0, 0],
                  [0, np.cos(np.radians(degree)), -np.sin(np.radians(degree))],
                  [0, np.sin(np.radians(degree)), np.cos(np.radians(degree))]]),
        'Y': np.array([[np.cos(np.radians(degree)), 0, np.sin(np.radians(degree))],
                  [0, 1, 0],
                  [-np.sin(np.radians(degree)), 0, np.cos(np.radians(degree))]]),
        'Z': np.array([[np.cos(np.radians(degree)), -np.sin(np.radians(degree)), 0],
                  [np.sin(np.radians(degree)), np.cos(np.radians(degree)), 0],
                  [0, 0, 1]])}
    newData = np.dot(data, xyzArray[axis])
    return newData

class UniHSI_PartNet_Train(humanoid_amp_task.HumanoidAMPTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        num_scenes = cfg["env"]["numScenes"] 
        self.num_scenes_col = num_scenes
        self.num_scenes_row = num_scenes
        self.env_array = torch.arange(0, cfg["env"]["numEnvs"], device=device_type, dtype=torch.float)
        self.spacing = cfg["env"]["envSpacing"]
        sceneplan_path = cfg['objFile']
        with open(sceneplan_path) as f:
            self.sceneplan = json.load(f)
        strike_body_names = cfg["env"]["strikeBodyNames"]
        self.plan_items = self.sceneplan


        self.joint_num = len(strike_body_names)
        self.local_scale = 9
        self.local_interval = 0.2

        self.max_step_pool_number = 30

        self.joint_name = ["pelvis", "left_hip", "left_knee", "left_foot", "right_hip", "right_knee", "right_foot", "torso", 
            "head", "left_shoulder", "left_elbow", "left_hand", "right_shoulder", "right_elbow", "right_hand"]
        self.joint_mapping = {"pelvis":0, "left_hip":1, "left_knee":2, "left_foot":3, "right_hip":4, "right_knee":5, "right_foot":6, "torso":7, 
            "head":8, "left_shoulder":9, "left_elbow":10, "left_hand":11, "right_shoulder":12, "right_elbow":13, "right_hand":14}
        
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._tar_dist_min = 0.5
        self._tar_dist_max = 10.0
        self._near_dist = 1.5
        self._near_prob = 0.5

        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        
        self._strike_body_ids = self._build_strike_body_ids_tensor(self.envs[0], self.humanoid_handles[0], strike_body_names)

        self.contact_type = torch.zeros([self.num_envs, self.joint_num], device=self.device, dtype=torch.bool)
        self.contact_valid = torch.zeros([self.num_envs, self.joint_num], device=self.device, dtype=torch.bool)

        self.joint_diff_buff = torch.ones([self.num_envs, self.joint_num], device=self.device, dtype=torch.float)
        self.location_diff_buf = torch.ones([self.num_envs], device=self.device, dtype=torch.float)
        self.joint_idx_buff = torch.ones([self.num_envs, self.joint_num], device=self.device, dtype=torch.long)
        self.tar_dir = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
    
        self.envs_idx = torch.arange(self.num_envs).to(self.device)

        self.pelvis2torso = torch.tensor([0,0,0.236151]).to(self.device)[None].repeat(self.num_envs, 1)
        self.pelvis2torso[:, 2] += 0.15
        self.torso2head = torch.tensor([0, 0, 0.223894]).to(self.device)[None].repeat(self.num_envs, 1)
        self.torso2head[:, 2] += 0.15
        self.new_rigid_body_pos = torch.ones([self.num_envs, 15, 3], device=self.device, dtype=torch.float)

        self.step_mode = torch.zeros([self.num_envs], device=self.device, dtype=torch.long)
        self.change_obj = torch.zeros([self.num_envs], device=self.device, dtype=torch.bool)
        self.stand_point_choice = torch.zeros([self.num_envs], device=self.device, dtype=torch.long)

        self.big_force = torch.zeros([self.num_envs], device=self.device, dtype=torch.bool)
        
        self.still = torch.zeros([self.num_envs], device=self.device, dtype=torch.bool)
        self.still_buf = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)

        return

    def _create_ground_plane(self):
        self._create_mesh_ground()
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _create_mesh_ground(self):
        self.plan_number = len(self.sceneplan)
        min_mesh_dict = self._load_mesh()
        pcd_list = self._load_pcd(min_mesh_dict)

        self._get_pcd_parts(pcd_list)

        _x = np.arange(0, self.local_scale)
        _y = np.arange(0, self.local_scale)
        _xx, _yy = np.meshgrid(_x, _y)
        x, y = _xx.ravel(), _yy.ravel()
        mesh_grid = np.stack([x, y, y], axis=-1) # 3rd dim meaningless
        self.mesh_pos = torch.from_numpy(mesh_grid).to(self.device) * self.local_interval
        self.humanoid_in_mesh = torch.tensor([self.local_interval*(self.local_scale-1)/4, self.local_interval*(self.local_scale-1)/2, 0]).to(self.device)

    def process_contact(self, label_dict, pcd, obj, steps, step_number, plan_id):

        direction_mapping = {
            "up":[0,0,1],
            "down":[0,0,-1],
            "left":[0,1,0],
            "right":[0,-1,0],
            "front":[1,0,0],
            "back":[-1,0,0],
            "none":[0,0,0]
        }
        for step_idx, step in enumerate(steps):
            contact_type_step = np.zeros(15)
            contact_valid_step = np.zeros(15)
            joint_pairs = np.zeros(15)
            joint_pairs_valid = np.zeros(15)
            contact_direction_step = np.zeros((15,3))

            for pair in step:
                obj_id = pair[0][-3:]
                label = label_dict[obj_id]['label']
                label_mapping = label_dict[obj_id]['label_mapping']
                if 'stand_point' in obj[obj_id].keys():
                    stand_point = obj[obj_id]['stand_point']
                else:
                    stand_point = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
                obj_pcd = pcd[int(obj_id)*10000:(int(obj_id)+1)*10000]
                if pair[1] != 'none' and pair[1] not in self.joint_name:
                    part_pcd = self._get_obj_parts(pair[0], [pair[1]], label_mapping, label, obj_pcd, stand_point)
                    joint_number = self.joint_mapping[pair[2]]
                    self.obj_pcd_buffer[plan_id, step_idx, joint_number] = part_pcd[0]
                    contact_type_step[joint_number] = 1 if pair[3] == 'contact' else 0
                    contact_valid_step[joint_number] = 1
                    contact_direction_step[joint_number] = direction_mapping[pair[4]]
                elif pair[1] in self.joint_name:
                    joint_number = self.joint_mapping[pair[2]]
                    target_joint_number = self.joint_mapping[pair[1]]
                    joint_pairs[joint_number] = target_joint_number
                    joint_pairs_valid[joint_number] = 1
                    contact_direction_step[joint_number] = direction_mapping[pair[4]]
            
            self.scene_stand_point[plan_id, step_idx] = torch.tensor(stand_point).float()
            self.contact_type_step[plan_id, step_idx] = torch.tensor(contact_type_step)
            self.contact_valid_step[plan_id, step_idx] = torch.tensor(contact_valid_step)
            self.contact_direction_step[plan_id, step_idx] = torch.tensor(contact_direction_step) 
            self.joint_pairs[plan_id, step_idx] = torch.tensor(joint_pairs)
            self.joint_pairs_valid[plan_id, step_idx] = torch.tensor(joint_pairs_valid)

    def _load_mesh(self):

        mesh_vertices_list = []
        mesh_triangles_list = []

        min_mesh_dict = dict()
        for plans in self.plan_items:
            plan = self.plan_items[plans]
            objs = plan['obj']
            min_mesh_dict[plans] = dict()

            l = 0
            pn = 0
            mesh_vertices = np.zeros([0, 3]).astype(np.float32)
            mesh_triangles = np.zeros([0, 3]).astype(np.uint32)
            for obj_id in objs:
                obj = objs[obj_id]
                pid = obj['id']
                mesh = o3d.io.read_triangle_mesh('data/partnet/'+pid+'/models/model_normalized.obj')
                for r in obj['rotate']:
                    R = mesh.get_rotation_matrix_from_xyz(r)
                    mesh.rotate(R, center=(0, 0, 0))
                mesh.scale(obj['scale'], center=mesh.get_center())
                mesh_vertices_single = np.asarray(mesh.vertices).astype(np.float32())
                mesh.translate((0,0,-mesh_vertices_single[:, 2].min()))
                mesh.translate(obj['transfer']) #  not collision with init human
                # mesh.translate(obj['multi_obj_offset'])
                mesh_vertices_single = np.asarray(mesh.vertices).astype(np.float32())
                mesh_triangles_single = np.asarray(mesh.triangles).astype(np.uint32)

                min_x_mesh, min_y_mesh, min_z_mesh = mesh_vertices_single[:,0].min(), mesh_vertices_single[:,1].min(), mesh_vertices_single[:,2].min()
                min_mesh_dict[plans][obj_id] = [min_x_mesh, min_y_mesh, min_z_mesh]

                mesh_vertices = np.concatenate([mesh_vertices, mesh_vertices_single], axis=0)
                mesh_triangles = np.concatenate([mesh_triangles, mesh_triangles_single], axis=0)
                mesh_triangles[l:] += pn
                l = mesh_triangles.shape[0]
                pn = mesh_vertices.shape[0]

            mesh_vertices_list.append(mesh_vertices)
            mesh_triangles_list.append(mesh_triangles)

        

        obj_idx = np.random.randint(0, self.plan_number, (self.num_scenes_row, self.num_scenes_col))
        obj_rotate = np.random.rand(self.num_scenes_row, self.num_scenes_col) * 360.0
        obj_rotate_matrix = np.array([[np.cos(np.radians(obj_rotate)), -np.sin(np.radians(obj_rotate)), obj_rotate*0],
                                    [np.sin(np.radians(obj_rotate)), np.cos(np.radians(obj_rotate)), obj_rotate*0],
                                    [obj_rotate*0, obj_rotate*0, obj_rotate*0+1]])

        self.obj_idx = torch.from_numpy(obj_idx).to(self.device)
        self.obj_rotate_matrix = torch.from_numpy(obj_rotate_matrix).to(self.device)

        dist_max = 2
        dist_min = 1
        rand_dist_x = (dist_max - dist_min) * np.random.rand(self.num_scenes_row, self.num_scenes_col) + dist_min
        rand_dist_y = (dist_max - dist_min) * np.random.rand(self.num_scenes_row, self.num_scenes_col) + dist_min
        rand_dist_x[0] = 0
        rand_dist_y[0] = 0
        self.rand_dist_x = torch.from_numpy(rand_dist_x).to(self.device)
        self.rand_dist_y = torch.from_numpy(rand_dist_y).to(self.device)
        rand_dist_z = np.random.rand(self.num_scenes_row, self.num_scenes_col) * 1.2 - 0.6
        change_height = 0
        rand_dist_z = rand_dist_z * change_height
        self.rand_dist_z = torch.from_numpy(rand_dist_z).to(self.device)


        scene_idx = np.random.randint(0, self.plan_number, (self.num_scenes_row, self.num_scenes_col))
        self.scene_idx = torch.from_numpy(scene_idx).to(self.device)

        for i in range(self.num_scenes_row):
            for j in range(self.num_scenes_col):
                mesh_vertices = mesh_vertices_list[self.scene_idx[i,j]]
                mesh_triangles = mesh_triangles_list[self.scene_idx[i,j]]
                mesh_vertices_offset = mesh_vertices.copy()
                mesh_vertices_offset = VaryPoint(mesh_vertices_offset, 'Z', obj_rotate[i,j]).astype(np.float32)
                mesh_vertices_offset[:, 0] += self.spacing *2 * i + rand_dist_x[i, j]
                mesh_vertices_offset[:, 1] += self.spacing *2 * j + rand_dist_y[i, j]
                mesh_vertices_offset[:, 2] += rand_dist_z[i, j]

                tm_params = gymapi.TriangleMeshParams()
                tm_params.nb_vertices = mesh_vertices_offset.shape[0]
                tm_params.nb_triangles = mesh_triangles.shape[0]
                self.gym.add_triangle_mesh(self.sim, mesh_vertices_offset.flatten(order='C'),
                                        mesh_triangles.flatten(order='C'),
                                        tm_params)
        
        return min_mesh_dict
    
    def _load_pcd(self, min_mesh_dict):

        trans_mat_path = 'data/partnet/chair_table_storagefurniture_bed_shapenetv1_to_partnet_alignment.json'
        with open(trans_mat_path, 'r') as fcc_file:
            trans_mat = fcc_file.read()
        trans_mat = json.loads(trans_mat)

        pcds = dict()
        for plans in self.plan_items:
            plan = self.plan_items[plans]
            objs = plan['obj']

            pcd_multi = []
            for obj_id in objs:
                obj = objs[obj_id]
                pid = obj['id']
                pcd = o3d.io.read_point_cloud("data/partnet/"+pid+"/point_sample/sample-points-all-pts-label-10000.ply")

                if pid == '11570' or pid == "11873" or pid == "4376" or pid == "5861":
                    pcd.scale(0.5, center=pcd.get_center())
                else:
                    matrix = np.array(trans_mat[pid]['transmat']).reshape(4,4)
                    tmp = matrix[0].copy()
                    matrix[0] = matrix[2]
                    matrix[2] = tmp
                    matrix = np.linalg.inv(matrix)
                    pcd.transform(matrix)

                for r in obj['rotate']:
                    R = pcd.get_rotation_matrix_from_xyz(r)
                    pcd.rotate(R, center=(0, 0, 0))
                pcd.scale(obj['scale'], center=pcd.get_center())

                if pid == '11570':
                    R = pcd.get_rotation_matrix_from_xyz((0, 0, -np.pi))
                    pcd.rotate(R, center=(0, 0, 0))
                

                pcd = np.asarray(pcd.points).astype(np.float32())

                max_y = pcd[:, 1].max()
                pcd[:, 1] = max_y - pcd[:, 1] # flip

                min_x_pcd, min_y_pcd, min_z_pcd = pcd[:,0].min(), pcd[:,1].min(), pcd[:,2].min()
                min_x_mesh, min_y_mesh, min_z_mesh = min_mesh_dict[plans][obj_id]
                pcd[:,0] += min_x_mesh-min_x_pcd 
                pcd[:,1] += min_y_mesh-min_y_pcd 
                pcd[:,2] += min_z_mesh-min_z_pcd
                pcd_multi.append(pcd)
            pcd_multi = np.concatenate(pcd_multi, axis=0)

            pcds[plans] = pcd_multi
        
        return pcds

    def _get_pcd_parts(self, pcd_list):
        # self.valid_joints_mask = torch.zeros([self.obj_number, 2, self.joint_num], dtype=bool, device=self.device) # hard code

        self.height_map = []
        self.obj_pcd_buffer = torch.zeros([self.plan_number, self.max_step_pool_number, 15, 200, 3])
        self.contact_type_step = torch.zeros([self.plan_number, self.max_step_pool_number, 15])
        self.contact_valid_step = torch.zeros([self.plan_number, self.max_step_pool_number, 15])
        self.contact_direction_step = torch.zeros([self.plan_number, self.max_step_pool_number, 15, 3])
        self.joint_pairs = torch.zeros([self.plan_number, self.max_step_pool_number, 15])
        self.joint_pairs_valid = torch.zeros([self.plan_number, self.max_step_pool_number, 15])        
        self.scene_stand_point = torch.zeros([self.plan_number, self.max_step_pool_number, 4, 3])        
        self.max_steps = torch.zeros(self.plan_number).int()
        self.contact_pairs = []

        for idx, plan_id in enumerate(self.plan_items):
            obj = self.plan_items[plan_id]['obj']
            contact_pairs = self.plan_items[plan_id]['contact_pairs']
            pcd = pcd_list[plan_id]
            step_number = len(contact_pairs)

            label_dict = dict()
            for obj_item in obj:
                obj_id = obj[obj_item]['id']
                label, label_mapping = self._get_part_labels(obj_id)
                label_dict[obj_item] = {'label': label, 'label_mapping': label_mapping}

            self.process_contact(label_dict, pcd, obj, contact_pairs, step_number, idx)
            heigh_pcd = get_height_map(pcd, HEIGHT_MAP_DIM=20)
            self.height_map.append(torch.from_numpy(heigh_pcd).to(self.device))
            self.max_steps[idx] = step_number
            self.contact_pairs.append(contact_pairs)


            

        self.height_map = torch.stack(self.height_map,0).to(self.device)
        self.scene_stand_point =  self.scene_stand_point.to(self.device)
        self.contact_type_step = self.contact_type_step.to(self.device).bool()
        self.contact_valid_step = self.contact_valid_step.to(self.device).bool()
        self.contact_direction_step = self.contact_direction_step.to(self.device).long()
        self.joint_pairs = self.joint_pairs.to(self.device).long()
        self.joint_pairs_valid = self.joint_pairs_valid.to(self.device).bool()
        self.contact_valid_step = self.contact_valid_step | self.joint_pairs_valid  # TODO add joint contact type
        self.contact_type_step = self.contact_type_step | self.joint_pairs_valid
        self.obj_pcd_buffer = self.obj_pcd_buffer.to(self.device)
        self.max_steps = self.max_steps.to(self.device)

    def _get_part_labels(self, partnet_id):

        if not isinstance(partnet_id, list):
            partnet_id = [partnet_id]

        result_dict_full = dict()
        offset = 0
        labels = []
        for pid in partnet_id:
            label_file = "data/partnet/"+pid+"/point_sample/sample-points-all-label-10000.txt"
            label = load_label(label_file) + offset
            result_file = "data/partnet/"+pid+"/result.json"
            with open(result_file, 'r') as fcc_file:
                result_file = fcc_file.read()
            result = json.loads(result_file)
            result_dict = dict()
            get_leaf_node(result_dict, result, offset)
            result_dict_full.update(result_dict)
            offset += 100 # hard code
            labels.append(label)
        labels = np.concatenate(labels, axis=0)
        
        return labels, result_dict_full

    def _get_obj_parts(self, object, contact_parts, label_mapping, label, pcd, stand_point):

        max_x, min_x, max_y, min_y = pcd[:, 0].max(), pcd[:, 0].min(), pcd[:, 1].max(), pcd[:, 1].min()   

        obj_pcd_buffer = []
        for p in contact_parts:
            if 'floor' in p:
                out_part_pcd = torch.rand(30000,3).cuda()
                out_part_pcd[:,0] = out_part_pcd[:,0] * ((max_x+0.5)-(min_x-0.5)) + min_x-0.5
                out_part_pcd[:,1] = out_part_pcd[:,1] * ((max_y+0.5)-(min_y-0.5)) + min_y-0.5
                out_part_pcd[:,2] = out_part_pcd[:,2] * 0.08
                if object[:3] == "bed":
                    mask = (out_part_pcd[:,1]>max_y+0.2)
                elif  object[:3] == "chair":
                    mask = (out_part_pcd[:,0]>max_x)
                else:
                    mask = ((out_part_pcd[:,0]>max_x) | (out_part_pcd[:,0]<min_x)) | ((out_part_pcd[:,1]>max_y) | (out_part_pcd[:,1]<min_y))
                out_part_pcd = out_part_pcd[mask]
            else:
                idx = label_mapping[p]
                part_pcd = pcd[label==idx]
                max_x, min_x, max_y, min_y, max_z, min_z = part_pcd[:,0].max(), part_pcd[:,0].min(), part_pcd[:,1].max(), part_pcd[:,1].min(), part_pcd[:,2].max(), part_pcd[:,2].min()
            
                out_part_pcd = part_pcd[(part_pcd[:,0] <= max((max_x-0.2), (max_x-min_x)/5*4+min_x)) & (part_pcd[:,0] >= min(min_x+0.2, (max_x-min_x)/5*1+min_x)) &
                                    (part_pcd[:,1] <= max((max_y-0.2), (max_y-min_y)/5*4+min_y)) & (part_pcd[:,1] >= min(min_y+0.2, (max_y-min_y)/5*1+min_y))] # filter edge
                out_part_pcd = torch.from_numpy(out_part_pcd).to(self.device)
            idx = farthest_point_sample(out_part_pcd[None], 200)
            out_part_pcd = out_part_pcd[idx]
            obj_pcd_buffer.append(out_part_pcd)
        obj_pcd_buffer = torch.cat(obj_pcd_buffer, 0)

        return obj_pcd_buffer

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 15 + self.joint_num * 2 + self.joint_num * 3  + self.local_scale*self.local_scale + self.joint_num*3
        return obs_size
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        self.spacing = spacing # env==2,3 have stupid bug

        self.env_scene_idx_row = (self.env_array % num_per_row % self.num_scenes_row).long()
        self.env_scene_idx_col = (self.env_array // num_per_row % self.num_scenes_col).long()
        self.scene_for_env = self.scene_idx[self.env_scene_idx_row, self.env_scene_idx_col]

        self.x_offset = (self.env_array % num_per_row % self.num_scenes_row) * spacing * 2 - (self.env_array % num_per_row) * spacing * 2
        self.y_offset = (self.env_array // num_per_row % self.num_scenes_col) * spacing * 2- (self.env_array // num_per_row) * spacing * 2

        # [num_envs, num_obj, num_part_sequence, num_pts. 3]
        self.envs_obj_pcd_buffer = self.obj_pcd_buffer.new_zeros([self.num_envs, self.obj_pcd_buffer.shape[2], self.obj_pcd_buffer.shape[3], self.obj_pcd_buffer.shape[4]])
        
        # self.envs_heightmap = self.height_map[self.scene_for_env].float()

        self.obj_rotate_matrix = self.obj_rotate_matrix.permute(2,3,0,1).float()
        self.scene_stand_point = torch.einsum("nmoe,neg->nmog", self.scene_stand_point[self.scene_for_env], self.obj_rotate_matrix[self.env_scene_idx_row, self.env_scene_idx_col])
        self.scene_stand_point[..., 0] += self.x_offset[:,None,None] + self.rand_dist_x[self.env_scene_idx_row, self.env_scene_idx_col][:,None,None]
        self.scene_stand_point[..., 1] += self.y_offset[:,None,None] + self.rand_dist_y[self.env_scene_idx_row, self.env_scene_idx_col][:,None,None]     
        
        self.envs_heightmap = self.height_map[self.scene_for_env].float()
        self.envs_heightmap[..., 0] += self.rand_dist_x[self.env_scene_idx_row, self.env_scene_idx_col][..., None]
        self.envs_heightmap[..., 1] += self.rand_dist_y[self.env_scene_idx_row, self.env_scene_idx_col][..., None]
        self.envs_heightmap[..., 2] += self.rand_dist_z[self.env_scene_idx_row, self.env_scene_idx_col][..., None]
        self.envs_heightmap = torch.einsum("nae,neg->nag", self.envs_heightmap, self.obj_rotate_matrix[self.env_scene_idx_row, self.env_scene_idx_col])
        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _build_strike_body_ids_tensor(self, env_ptr, actor_handle, body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _reset_actors(self, env_ids):
        if len(env_ids) > 0:
            success = (self.location_diff_buf[env_ids] < 0.1) & ~self.big_force[env_ids]

        self._reset_target(env_ids, success)

    def _reset_target(self, env_ids, success):

        contact_type_steps = self.contact_type_step[self.scene_for_env, self.step_mode]
        contact_valid_steps = self.contact_valid_step[self.scene_for_env, self.step_mode]
        fulfill = ((contact_valid_steps & \
                     (((contact_type_steps) & (self.joint_diff_buff < 0.1)) | (((~contact_type_steps) & (self.joint_diff_buff >= 0.05))))) \
                        | (~contact_valid_steps))[env_ids] & (success[:, None]) # need add contact direction
        fulfill = torch.all(fulfill, dim=-1)

        self.step_mode[env_ids[fulfill]] += 1

        max_step = self.step_mode[env_ids] == self.max_steps[self.scene_for_env][env_ids]


        reset = ~fulfill | max_step
        super()._reset_actors(env_ids[reset])

        self.still_buf[env_ids[reset|fulfill]] = 0

        rand_rot_theta = 2 * np.pi * torch.rand([self.num_envs], device=self.device)
        axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        rand_rot = quat_from_angle_axis(rand_rot_theta, axis)
        self._humanoid_root_states[env_ids[reset], 3:7] = rand_rot[env_ids[reset]]

        dist_max = 4
        dist_min = 2
        rand_dist_y = (dist_max - dist_min) * torch.rand([self.num_envs], device=self.device) + dist_min
        rand_dist_x = (dist_max - dist_min) * torch.rand([self.num_envs], device=self.device) + dist_min
        x_sign = torch.from_numpy(np.random.choice((-1, 1), [self.num_envs])).to(self.device)
        y_sign = torch.from_numpy(np.random.choice((-1, 1), [self.num_envs])).to(self.device)
        self._humanoid_root_states[env_ids[reset], 0] += self.x_offset[env_ids[reset]] + x_sign[env_ids[reset]] * rand_dist_x[env_ids[reset]]
        self._humanoid_root_states[env_ids[reset], 1] += self.y_offset[env_ids[reset]] + y_sign[env_ids[reset]] * rand_dist_y[env_ids[reset]] - 2
        self.step_mode[env_ids[reset]] = 0

        stand_point_choice = torch.from_numpy(np.random.choice((0,1,2,3), [self.num_envs])).to(self.device)
        self.stand_point_choice[env_ids[reset]] = stand_point_choice[env_ids[reset]]

        self.contact_type = self.contact_type_step[self.scene_for_env, self.step_mode]
        self.contact_valid = self.contact_valid_step[self.scene_for_env, self.step_mode]
        self.contact_direction = self.contact_direction_step[self.scene_for_env, self.step_mode]

        self.stand_point = self.scene_stand_point[range(len(self.step_mode)), self.step_mode, self.stand_point_choice]

        self.envs_obj_pcd_buffer[env_ids] = self.obj_pcd_buffer[self.scene_for_env[env_ids], self.step_mode[env_ids]]
        self.envs_obj_pcd_buffer[env_ids] = torch.einsum("nmoe,neg->nmog", self.envs_obj_pcd_buffer[env_ids], self.obj_rotate_matrix[self.env_scene_idx_row, self.env_scene_idx_col][env_ids])
        self.envs_obj_pcd_buffer[env_ids, ..., 0] += self.x_offset[:, None, None][env_ids] + self.rand_dist_x[self.env_scene_idx_row, self.env_scene_idx_col][..., None, None][env_ids]
        self.envs_obj_pcd_buffer[env_ids, ..., 1] += self.y_offset[:, None, None][env_ids] + self.rand_dist_y[self.env_scene_idx_row, self.env_scene_idx_col][..., None, None][env_ids]
        self.envs_obj_pcd_buffer[env_ids, ..., 2] += self.rand_dist_z[self.env_scene_idx_row, self.env_scene_idx_col][..., None, None][env_ids]

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        return
    
    def _compute_task_obs(self, env_ids=None):

        pcd_buffer = []
        self.new_rigid_body_pos = self._rigid_body_pos.clone()

        head = quat_rotate(self._rigid_body_rot[:, 1], self.torso2head)
        head += self._rigid_body_pos[:, 1]
        self.new_rigid_body_pos[:, 2] = head

        torso = quat_rotate(self._rigid_body_rot[:, 0], self.pelvis2torso)
        torso += self._rigid_body_pos[:, 0]
        self.new_rigid_body_pos[:, 1] = torso
        joint_pos_buffer = []
        if (env_ids is None):
            root_states = self._humanoid_root_states
            tar_pos = self.stand_point
            pcd_buffer = self.envs_obj_pcd_buffer[self.envs_idx]
            env_num, joint_num, point_num, point_dim = pcd_buffer.shape
            pcd_buffer = pcd_buffer.view(-1, point_num, point_dim)
            pcd_buffer = pcd_buffer[range(pcd_buffer.shape[0]), self.joint_idx_buff.view(-1)]
            pcd_buffer = pcd_buffer.reshape(env_num, joint_num, point_dim)
            joint_pos_buffer = self.new_rigid_body_pos[:, self._strike_body_ids]

            joint_contact_choice = self.joint_pairs[self.scene_for_env, self.step_mode]
            valid_joint_contact_choice = self.joint_pairs_valid[self.scene_for_env, self.step_mode]
            joints_contact = joint_pos_buffer.view(-1, 3)[joint_contact_choice.view(-1)].clone()
            pcd_buffer_view = pcd_buffer.view(-1, 3)
            pcd_buffer_view[valid_joint_contact_choice.view(-1)] = joints_contact[valid_joint_contact_choice.view(-1)]

            height_map = self.envs_heightmap

            contact_type = self.contact_type
            contact_valid = self.contact_valid
            contact_direction = self.contact_direction

            origin_root_pos = root_states[:, :3].clone()
            origin_root_pos[:, 0] = origin_root_pos[:, 0] - self.x_offset
            origin_root_pos[:, 1] = origin_root_pos[:, 1] - self.y_offset
            tar_dir = self.tar_dir
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_pos = self.stand_point[env_ids]
            pcd_buffer = self.envs_obj_pcd_buffer[self.envs_idx]
            env_num, joint_num, point_num, point_dim = pcd_buffer.shape
            pcd_buffer = pcd_buffer.view(-1, point_num, point_dim)
            pcd_buffer = pcd_buffer[range(pcd_buffer.shape[0]), self.joint_idx_buff.view(-1)]
            pcd_buffer = pcd_buffer.reshape(env_num, joint_num, point_dim)
            pcd_buffer = pcd_buffer[env_ids]
            joint_pos_buffer = self.new_rigid_body_pos[:, self._strike_body_ids][env_ids]

            joint_contact_choice = self.joint_pairs[self.scene_for_env, self.step_mode][env_ids]
            valid_joint_contact_choice = self.joint_pairs_valid[self.scene_for_env, self.step_mode][env_ids]
            joints_contact = joint_pos_buffer.view(-1, 3)[joint_contact_choice.view(-1)].clone()
            pcd_buffer_view = pcd_buffer.view(-1, 3)
            pcd_buffer_view[valid_joint_contact_choice.view(-1)] = joints_contact[valid_joint_contact_choice.view(-1)]

            height_map = self.envs_heightmap[env_ids]
            contact_type = self.contact_type[env_ids]
            contact_valid = self.contact_valid[env_ids]
            contact_direction = self.contact_direction[env_ids]
        
            origin_root_pos = root_states[:, :3].clone()
            origin_root_pos[:, 0] = origin_root_pos[:, 0] - self.x_offset[env_ids]
            origin_root_pos[:, 1] = origin_root_pos[:, 1] - self.y_offset[env_ids]
            tar_dir = self.tar_dir[env_ids]

        tar_rot = root_states.new_zeros([root_states.shape[0],4])
        tar_rot[:, 3] = 1
        tar_vel = root_states.new_zeros([root_states.shape[0],3])
        tar_ang_vel = root_states.new_zeros([root_states.shape[0],3])
        obs, self.local_height_map, self.rotated_mesh_pos = compute_strike_observations(root_states, tar_pos, joint_pos_buffer, pcd_buffer, tar_rot, tar_vel, tar_ang_vel, contact_type, contact_valid, contact_direction,
                                                                                 self.humanoid_in_mesh, origin_root_pos, self.local_scale, height_map, self.mesh_pos, tar_dir)
        
        char_root_state = self._humanoid_root_states
        target = self.stand_point

        pcd_buffer = self.envs_obj_pcd_buffer[self.envs_idx]
        joint_pos_buffer = self.new_rigid_body_pos[..., self._strike_body_ids, :]

        joint_contact_choice = self.joint_pairs[self.scene_for_env, self.step_mode]
        valid_joint_contact_choice = self.joint_pairs_valid[self.scene_for_env, self.step_mode]
        joints_contact = joint_pos_buffer.view(-1, 3)[joint_contact_choice.view(-1)].clone()
        pcd_buffer_view = pcd_buffer.view(-1, pcd_buffer.shape[-2], 3)
        pcd_buffer_view[valid_joint_contact_choice.view(-1)] = joints_contact[valid_joint_contact_choice.view(-1)][:, None]
        pcd_buffer = pcd_buffer_view.reshape(pcd_buffer.shape)

        self.rew_buf[:], self.location_diff_buf[:], self.joint_diff_buff[:], self.joint_idx_buff[:], self.tar_dir[:, :2] = compute_contact_reward(target, char_root_state,
                                                                                 pcd_buffer, joint_pos_buffer,
                                                                                 self._prev_root_pos,
                                                                                 self.dt, self.contact_type, self.contact_valid, self.contact_direction)
        
        return obs

    def _compute_reset(self):
        # calcute reset conditions
        success = (self.location_diff_buf < 0.1) & ~self.big_force
        contact_type_steps = self.contact_type_step[self.scene_for_env, self.step_mode]
        contact_valid_steps = self.contact_valid_step[self.scene_for_env, self.step_mode]
        fulfill = ((contact_valid_steps & \
                     (((contact_type_steps) & (self.joint_diff_buff < 0.3)) | (((~contact_type_steps) & (self.joint_diff_buff >= 0.1))))) \
                        | (~contact_valid_steps))& (success[:, None])
        fulfill = torch.all(fulfill, dim=-1)
        self.still = self.still_buf>10 & fulfill
        self.big_force = (self._contact_forces.abs()>10000).sum((-2,-1))>0

        self.reset_buf[:], self._terminate_buf[:], self.still_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                           self._contact_forces, self._contact_body_ids,
                                                           self._rigid_body_pos,
                                                           self._strike_body_ids, self.max_episode_length,
                                                           self._enable_early_termination, self._termination_heights, self._rigid_body_vel, self.still_buf, fulfill, self.big_force)
        return

    def _draw_task(self):
        
        self.gym.clear_lines(self.viewer)
        cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        starts = self.new_rigid_body_pos[0][self._strike_body_ids][self.contact_valid[0]]
        ends = self.envs_obj_pcd_buffer[0][range(15), self.joint_idx_buff[0]][self.contact_valid[0]]

        joint_pos_buffer = self.new_rigid_body_pos[:, self._strike_body_ids][0]
        joint_contact_choice = self.joint_pairs[self.scene_for_env, self.step_mode][0]
        valid_joint_contact_choice = self.joint_pairs_valid[self.scene_for_env, self.step_mode][0]
        joints_contact = joint_pos_buffer.view(-1, 3)[joint_contact_choice.view(-1)].clone()
        ends[valid_joint_contact_choice.view(-1)[self.contact_valid[0]]] = joints_contact[valid_joint_contact_choice.view(-1)]
        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()
        # verts = verts[7:9]
        cols = cols.repeat(verts.shape[0], axis=0)

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts
            curr_verts = curr_verts.reshape(-1, 6)
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)


        starts = self._humanoid_root_states[..., 0:3]
        # ends = self.pcd_buffer[0].mean(1)
        ends = self.stand_point
        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

        return

#####################################################################
###=========================jit functions=========================###
#####################################################################

# @torch.jit.script
def compute_strike_observations(root_states, tar_pos, joint_pos_buffer, pcd_buffer, tar_rot, tar_vel, tar_ang_vel, contact_type, contact_valid, contact_direction, 
                                human_in_mesh, origin_root_pos, local_scale, height_map_pcd, mesh_pos, tar_dir):
    # type: (Tensor, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    local_tar_pos = tar_pos - root_pos
    local_tar_pos[..., -1] = tar_pos[..., -1]
    local_tar_pos = quat_rotate(heading_rot, local_tar_pos)
    local_tar_vel = quat_rotate(heading_rot, tar_vel)
    local_tar_ang_vel = quat_rotate(heading_rot, tar_ang_vel)

    local_tar_rot = quat_mul(heading_rot, tar_rot)
    local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)

    mesh_pos = mesh_pos[None] + (root_pos -human_in_mesh)[:, None]
    mesh_dist = (mesh_pos - root_pos[:, None]).reshape(-1, 3)
    heading_rot_for_height = root_rot[:, None].repeat(1, local_scale*local_scale, 1).reshape(-1, 4)
    heading_rot_for_height[:, :2] = 0
    heading_rot_norm = torch.sqrt(1 - heading_rot_for_height[:, 3]*heading_rot_for_height[:, 3])
    heading_rot_for_height[:, 2] = heading_rot_for_height[:, 2] * heading_rot_norm / torch.abs(heading_rot_for_height[:, 2])
    rotated_mesh_dist = quat_rotate(heading_rot_for_height, mesh_dist).reshape(-1, local_scale*local_scale, 3)
    rotated_mesh_pos = rotated_mesh_dist + root_pos[:, None]
    rotated_mesh_origin_pos = rotated_mesh_dist + origin_root_pos[:, None]
    dist = rotated_mesh_origin_pos[..., :2][:, None] - height_map_pcd[..., :2][:, :, None]
    sum_dist = torch.sum(dist * dist, dim=-1)
    shape = sum_dist.shape
    sum_dist = sum_dist.permute(1,0,2).reshape(shape[1], -1)
    min_idx = sum_dist.argmin(0)
    dist_min = sum_dist[min_idx, range(min_idx.shape[0])]
    valid_mask = dist_min < 0.05
    height_map = height_map_pcd.reshape(-1, 3)[min_idx, 2]
    height_map[~valid_mask] = 0.0
    height_map = height_map.reshape(shape[0], shape[2]).float()


    contact_direction = contact_direction.reshape(-1, 45)
    obs = torch.cat([local_tar_pos, local_tar_rot_obs, local_tar_vel, tar_dir, contact_type, contact_valid, height_map], dim=-1)

    local_target_pos = pcd_buffer - joint_pos_buffer
    local_target_pos_r = quat_rotate(heading_rot[:, None].repeat(1, local_target_pos.shape[1], 1).view(-1, 4), local_target_pos.view(-1, 3))
    local_target_pos_r = local_target_pos_r.reshape(local_target_pos.shape)
    local_target_pos_r *= contact_valid[..., None]
    obs = torch.cat([obs, local_target_pos_r.view(obs.shape[0], -1), contact_direction], dim=-1)

    return obs, height_map, rotated_mesh_pos

def compute_contact_reward(target, root_state, pcd_buffer, joint_pos_buffer, 
                           prev_root_pos, dt, contact_type, contact_valid, contact_direction):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, float) -> Tensor
    dist_threshold = 0.2

    pos_err_scale = 0.5
    vel_err_scale = 2.0
    near_pos_err_scale = 10

    tar_speed = 1.0
    
    root_pos = root_state[..., 0:3]
    root_rot = root_state[..., 3:7]

    contact_type = contact_type.float()

    near_pos_reward_buf = []
    min_pos_idx_buf = []
    near_pos_err_min_buf = []
    for i in range(pcd_buffer.shape[1]):
        near_pos_diff = pcd_buffer[:, i] - joint_pos_buffer[:, i][:, None]
        near_pos_err = torch.sum(near_pos_diff * near_pos_diff, dim=-1)
        near_pos_err_min, min_pos_idx = near_pos_err.min(-1)
        near_pos_reward = torch.exp(-near_pos_err_scale * near_pos_err_min)
        near_pos_reward_contact = near_pos_reward * contact_type[:,i] + (1-near_pos_reward) * (1-contact_type[:,i])

        reward_dir = torch.nn.functional.normalize(-near_pos_diff[range(near_pos_diff.shape[0]), min_pos_idx], dim=-1) * contact_direction[:,i]
        reward_dir = reward_dir.sum(-1)
        reward_dir[contact_direction[:,i].sum(1)==0] = 1
        reward_dir[reward_dir<0] = 0
        reward_dir[~contact_type[:,i].bool()] = 1
        contact_w = (1 - near_pos_reward_contact) / (2 - near_pos_reward_contact - reward_dir + 1e-4)
        dir_w = (1 - reward_dir) / (2 - near_pos_reward_contact - reward_dir + 1e-4)
        near_pos_err_min[reward_dir<0.5] += 1 # not fullfill dir
        near_pos_reward_contact = contact_w * near_pos_reward_contact + dir_w * reward_dir

        near_pos_reward_contact[~contact_valid[:, i]] = 1
        near_pos_reward_buf.append(near_pos_reward_contact)
        min_pos_idx_buf.append(min_pos_idx)
        near_pos_err_min_buf.append(near_pos_err_min)
    near_pos_err_min_buf = torch.stack(near_pos_err_min_buf, 1)
    near_pos_reward_buf = torch.stack(near_pos_reward_buf, 0)
    min_pos_idx_buf = torch.stack(min_pos_idx_buf, 1)
    near_pos_reward_w = (1 - near_pos_reward_buf) / (pcd_buffer.shape[1] - near_pos_reward_buf.sum(0) + 1e-4)

    near_pos_reward = (near_pos_reward_w * near_pos_reward_buf).sum(0)

    facing_target = (contact_valid.sum(-1) == 1) & (contact_valid[:, -1] | contact_valid[:, -4])
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir)
    
    target_pcd = pcd_buffer[range(pcd_buffer.shape[0]), -1, min_pos_idx_buf[:, -1]]
    pos_diff = target_pcd - root_pos
    tar_dir = torch.nn.functional.normalize(pos_diff[..., 0:2], dim=-1)
    tar_dir[~facing_target] = facing_dir[~facing_target][..., 0:2]
    obj_facing_reward = torch.sum(tar_dir * facing_dir[..., 0:2], dim=-1)
    obj_facing_reward = torch.clamp_min(obj_facing_reward, 0.0)

    obj_facing_reward_w = (1-obj_facing_reward) / (2-obj_facing_reward-near_pos_reward + 1e-4)
    near_pos_reward_w = (1-near_pos_reward) / (2-obj_facing_reward-near_pos_reward + 1e-4)
    
    near_pos_reward = obj_facing_reward_w * obj_facing_reward + near_pos_reward_w * near_pos_reward

    pos_diff = target - root_pos
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-pos_err_scale * pos_err)

    dist_mask = pos_err < dist_threshold

    tar_dir = torch.nn.functional.normalize(pos_diff[..., 0:2], dim=-1)
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = tar_speed - tar_dir_speed
    vel_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0
    vel_reward[dist_mask] = 1

    heading_rot = torch_utils.calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir)
    facing_err = torch.sum(tar_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)
    facing_reward[dist_mask] = 1

    pos_reward_w = (1-pos_reward) / (3-pos_reward-vel_reward-facing_reward + 1e-4)
    vel_reward_w = (1-vel_reward) / (3-pos_reward-vel_reward-facing_reward + 1e-4)
    face_reward_w = (1-facing_reward) / (3-pos_reward-vel_reward-facing_reward + 1e-4)

    far_reward = pos_reward_w * pos_reward + vel_reward_w * vel_reward + face_reward_w * facing_reward

    not_walking = contact_valid.sum(-1)>0

    # dist_mask = pos_err < dist_threshold
    reward = far_reward
    reward[not_walking] = near_pos_reward[not_walking]
    pos_err[not_walking] = 0 # once success, keep success
    # reward[dist_mask] = reward_near[dist_mask]

    return reward, pos_err, near_pos_err_min_buf, min_pos_idx_buf, tar_dir
    
# @torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos, strike_body_ids, max_episode_length,
                           enable_early_termination, termination_heights, _rigid_body_vel, still_buf, fulfill, big_force):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
    contact_force_threshold = 1.0
    
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        # tar_has_contact = torch.any(torch.abs(tar_contact_forces[..., 0:2]) > contact_force_threshold, dim=-1)
        #strike_body_force = contact_buf[:, strike_body_id, :]
        #strike_body_has_contact = torch.any(torch.abs(strike_body_force) > contact_force_threshold, dim=-1)
        nonstrike_body_force = masked_contact_buf
        nonstrike_body_force[:, strike_body_ids, :] = 0
        nonstrike_body_has_contact = torch.any(torch.abs(nonstrike_body_force) > contact_force_threshold, dim=-1)
        nonstrike_body_has_contact = torch.any(nonstrike_body_has_contact, dim=-1)

        # tar_fail = torch.logical_and(tar_has_contact, nonstrike_body_has_contact)
        
        # has_failed = torch.logical_or(has_fallen, tar_fail)
        has_failed = has_fallen

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_failed *= (progress_buf > 1)
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)
    
    still_buf[_rigid_body_vel.abs().sum(-1).max(-1)[0]<0.6] += 1
    still_buf[_rigid_body_vel.abs().sum(-1).max(-1)[0]>0.6] = 0
    # print(_rigid_body_vel.abs().sum(-1).max(-1))

    # print(still_buf)

    terminated = torch.where(big_force, torch.ones_like(reset_buf), terminated) # terminate when force is too big (could cause peneration)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)
    reset = torch.where((still_buf>10) & fulfill, torch.ones_like(reset_buf), reset)
    
    return reset, terminated, still_buf