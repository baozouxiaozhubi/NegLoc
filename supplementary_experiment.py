from datapreparation.kitti360pose.imports import Object3d
import numpy as np
import os.path as osp
from easydict import EasyDict
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
import time

from scipy.spatial.distance import cdist

from models.cell_retrieval import CellRetrievalNetwork
from models.cross_matcher import CrossMatch

from evaluation.args import parse_arguments
from evaluation.utils import calc_sample_accuracies, print_accuracies

from dataloading.kitti360pose.cells import Kitti360CoarseDataset, Kitti360CoarseDatasetMulti
from dataloading.kitti360pose.eval import Kitti360TopKDataset

from datapreparation.kitti360pose.utils import SCENE_NAMES_TEST, SCENE_NAMES_VAL, KNOWN_CLASS
from datapreparation.kitti360pose.utils import COLOR_NAMES as COLOR_NAMES_K360
from datapreparation.kitti360pose.drawing import plot_cells_and_poses_pred_and_gt

from training.coarse import eval_epoch as eval_epoch_retrieval

from PIL import Image

import torch_geometric.transforms as T
import tqdm

def z_score(tensor):
    #input：[Batch_size,Batch_size]
    mean = np.mean(tensor)
    std = np.std(tensor)
    normalized_tensor = (tensor - mean) / std
    return normalized_tensor

def euclidean_distance_2d(point1, point2):
    """
    计算二维空间中两个点之间的欧几里得距离。

    Args:
        point1 (tuple or list): 第一个点的坐标 (x1, y1)。
        point2 (tuple or list): 第二个点的坐标 (x2, y2)。

    Returns:
        float: 两点之间的欧几里得距离。
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    return np.linalg.norm(point1 - point2)
def eval_epoch(model, dataloader, args, return_encodings=False, return_distance=False,thresh_query=200 ,thresh_database=200):
    model.eval()  
    center=[-2686.3046,41.2897]
    accuracies = {k: [] for k in args.top_k}
    accuracies_close = {k: [] for k in args.top_k}

    cells_dataset = dataloader.dataset.get_cell_dataset()
    cells_dataloader = DataLoader(
        cells_dataset,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )
    cells_dict = {cell.id: cell for cell in cells_dataset.cells}
    cell_size = cells_dataset.cells[0].cell_size

    circle_mask_of_database=np.zeros(len(cells_dataset))

    cell_encodings_global_level=np.zeros((len(cells_dataset),model.embed_dim))
    cell_encodings_object_level=np.zeros((len(cells_dataset),args.object_size,model.embed_dim))
    cell_encodings_relation_level=np.zeros((len(cells_dataset),args.object_size,model.embed_dim))
    cell_encodings_object_level_mask=np.zeros((len(cells_dataset),args.object_size,1))
    cell_encodings_relation_level_mask=np.zeros((len(cells_dataset),args.object_size,1))

    db_cell_ids = np.zeros(len(cells_dataset), dtype="<U32")

    circle_mask_of_query=np.zeros(len(dataloader.dataset))

    text_encodings = np.zeros((len(dataloader.dataset), model.embed_dim))
    text_encodings_object_level=np.zeros((len(dataloader.dataset),args.num_mentioned,model.embed_dim))
    text_encodings_relation_level=np.zeros((len(dataloader.dataset),args.num_mentioned,model.embed_dim))
    query_cell_ids = np.zeros(len(dataloader.dataset), dtype="<U32")
    query_poses_w = np.array([pose.pose_w[0:2] for pose in dataloader.dataset.all_poses])

    # Encode the query side
    t0 = time.time()
    index_offset = 0
    for batch in tqdm.tqdm(dataloader):

        #filter out the query that is too far away from center
        pose_w_list=[data.pose_w[0:2].tolist() for data in batch["poses"]]
        if euclidean_distance_2d(center,pose_w_list)<thresh_query:
            circle_mask_of_query[index_offset : index_offset + len(batch["poses"])]=1
        else:
            index_offset+=len(batch["poses"])
            continue


        text_enc,text_enc_object_level,text_enc_relation_level = model.encode_text(batch["texts"])
        batch_size = len(text_enc)

        text_encodings[index_offset : index_offset + batch_size, :] = (
            text_enc.cpu().detach().numpy()
        )
        text_encodings_object_level[index_offset : index_offset + batch_size, : , :] = (
            text_enc_object_level.cpu().detach().numpy()
        )
        text_encodings_relation_level[index_offset : index_offset + batch_size, :,  :] = (
            text_enc_relation_level.cpu().detach().numpy()
        )
        query_cell_ids[index_offset : index_offset + batch_size] = np.array(batch["cell_ids"])
        index_offset += batch_size
        del text_enc,text_enc_object_level,text_enc_relation_level

    #filter out the query that is too far away from center
    text_encodings=text_encodings[circle_mask_of_query==1]
    text_encodings_object_level=text_encodings_object_level[circle_mask_of_query==1]
    text_encodings_relation_level=text_encodings_relation_level[circle_mask_of_query==1]
    query_cell_ids=query_cell_ids[circle_mask_of_query==1]

    print(f"Encoded {len(text_encodings)} query texts in {time.time() - t0:0.2f}.")

    # Encode the database side
    index_offset = 0
    for batch in cells_dataloader:

        #filter out the database that is too far away from center
        centers=[data.get_center()[0:2].tolist() for data in batch["cells"]]
        if euclidean_distance_2d(center,centers)<thresh_database:
            circle_mask_of_database[index_offset : index_offset + len(batch["cells"])]=1
        else:
            index_offset+=len(batch["cells"])
            continue
        cell_enc_global_level, cell_enc_object_level,cell_enc_relation_level,cell_enc_object_level_mask,cell_enc_relation_level_mask = model.encode_objects(batch["objects"], batch["object_points"])
        batch_size = len(cell_enc_object_level)

        cell_encodings_global_level[index_offset : index_offset + batch_size, : ] = (
            cell_enc_global_level.cpu().detach().numpy()
        )

        cell_encodings_object_level[index_offset : index_offset + batch_size, : , :] = (
            cell_enc_object_level.cpu().detach().numpy()
        )
        
        cell_encodings_relation_level[index_offset : index_offset + batch_size, :, :] = (
            cell_enc_relation_level.cpu().detach().numpy()
        )
        
        cell_encodings_object_level_mask[index_offset : index_offset + batch_size, :, :] = (
            cell_enc_object_level_mask.cpu().detach().numpy()
        )
        
        cell_encodings_relation_level_mask[index_offset : index_offset + batch_size, :, :] = (
            cell_enc_relation_level_mask.cpu().detach().numpy()
        )
        db_cell_ids[index_offset : index_offset + batch_size] = np.array(batch["cell_ids"])
        index_offset += batch_size
        del cell_enc_global_level,cell_enc_object_level,cell_enc_relation_level,cell_enc_object_level_mask,cell_enc_relation_level_mask
    
    #filter out the database that is too far away from center
    cell_encodings_global_level=cell_encodings_global_level[circle_mask_of_database==1]
    cell_encodings_object_level=cell_encodings_object_level[circle_mask_of_database==1]
    cell_encodings_relation_level=cell_encodings_relation_level[circle_mask_of_database==1]
    cell_encodings_object_level_mask=cell_encodings_object_level_mask[circle_mask_of_database==1]
    cell_encodings_relation_level_mask=cell_encodings_relation_level_mask[circle_mask_of_database==1]
    db_cell_ids=db_cell_ids[circle_mask_of_database==1]
    print(f"Encoded {len(cell_encodings_global_level)} cells in {time.time() - t0:0.2f}.")

    top_retrievals = {}  # {query_idx: top_cell_ids}
    if return_distance:
        dists_list = []
        scores_list = []
    for query_idx in range(len(text_encodings)):
        if args.ranking_loss != "triplet":  # Asserted above
            # similarity score of Global-level
            scores_1=cell_encodings_global_level[:] @ text_encodings[query_idx] 

            # similarity score of Object-level
            aa=torch.from_numpy(text_encodings_object_level[query_idx])
            bb=torch.from_numpy(cell_encodings_object_level)
            aa=aa.unsqueeze(0).expand(len(cell_encodings_object_level),args.num_mentioned,args.coarse_embed_dim)
            bb=bb.transpose(1,2)
            cc=torch.matmul(aa,bb)
            aa_mask=torch.ones((aa.size(1),1)).to(dtype=torch.float64).to('cuda')
            bb_mask=torch.from_numpy(cell_encodings_object_level_mask).transpose(1,2).to('cuda')
            scores_2_mask=torch.matmul(aa_mask,bb_mask)
            cc[scores_2_mask==0]=-100
            scores_2=cc.max(dim=-1)[0].mean(dim=-1)

            # similarity score of Relation-level
            aa=torch.from_numpy(text_encodings_relation_level[query_idx])
            bb=torch.from_numpy(cell_encodings_relation_level)
            aa=aa.unsqueeze(0).expand(len(cell_encodings_object_level),args.num_mentioned,args.coarse_embed_dim)
            bb=bb.transpose(1,2)
            cc=torch.matmul(aa,bb)
            aa_mask=torch.ones((aa.size(1),1)).to(dtype=torch.float64).to('cuda')
            bb_mask=torch.from_numpy(cell_encodings_relation_level_mask).transpose(1,2).to('cuda')
            scores_3_mask=torch.matmul(aa_mask,bb_mask)
            cc[scores_3_mask==0]=-100
            scores_3=cc.max(dim=-1)[0].mean(dim=-1)

            scores=1.0*z_score(scores_1)+1.0*z_score(np.array(scores_2))+1.0*z_score(np.array(scores_3))

            #assert len(scores) == len(dataloader.dataset.all_cells) 
            sorted_indices = np.argsort(-1.0 * scores)  # High -> low

        sorted_indices = sorted_indices[0 : np.max(args.top_k)]

        # Best-cell hit accuracy
        retrieved_cell_ids = db_cell_ids[sorted_indices]
        target_cell_id = query_cell_ids[query_idx]

        for k in args.top_k:
            accuracies[k].append(target_cell_id in retrieved_cell_ids[0:k])
        top_retrievals[query_idx] = retrieved_cell_ids

        # Close-by accuracy
        # CARE/TODO: can be wrong across scenes!
        target_pose_w = query_poses_w[query_idx]

        retrieved_cell_poses = [
            cells_dict[cell_id].get_center()[0:2] for cell_id in retrieved_cell_ids
        ]
        dists = np.linalg.norm(target_pose_w - retrieved_cell_poses, axis=1)
        if return_distance:
            dists_list.append(dists[0:max(args.top_k)])
            scores_list.append(scores[sorted_indices])
        for k in args.top_k:
            accuracies_close[k].append(np.any(dists[0:k] <= cell_size / 2))

    for k in args.top_k:
        accuracies[k] = np.mean(accuracies[k])
        accuracies_close[k] = np.mean(accuracies_close[k])

    
    return accuracies, accuracies_close, top_retrievals
    
