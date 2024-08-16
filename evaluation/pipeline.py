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

"""
TODO:
- Try to add num_matches*10 + sum(match_scores[correctly_matched])
"""


@torch.no_grad()
def run_coarse(model, dataloader, args):
    """Run text-to-cell retrieval to obtain the top-cells and coarse pose accuracies.

    Args:
        model: retrieval model
        dataloader: retrieval dataset
        args: global arguments

    Returns:
        [List]: retrievals as [(cell_indices_i_0, cell_indices_i_1, ...), (cell_indices_i+1, ...), ...] with i ∈ [0, len(poses)-1], j ∈ [0, max(top_k)-1]
        [Dict]: accuracies
    """
    model.eval()

    all_cells_dict = {cell.id: cell for cell in dataloader.dataset.all_cells}

        # Run retrieval model to obtain top-cells
        
    retrieval_accuracies, retrieval_accuracies_close, retrievals = eval_epoch_retrieval(
        model, dataloader, args
    )
    retrievals = [retrievals[idx] for idx in range(len(retrievals))]  # Dict -> list
    print("Retrieval Accs:")
    print(retrieval_accuracies)
    print("Retrieval Accs Close:")
    print(retrieval_accuracies_close)
    assert len(retrievals) == len(dataloader.dataset.all_poses)

    # Gather the accuracies for each sample
    accuracies = {k: {t: [] for t in args.threshs} for k in args.top_k}
    for i_sample in range(len(retrievals)):
        pose = dataloader.dataset.all_poses[i_sample]
        top_cells = [all_cells_dict[cell_id] for cell_id in retrievals[i_sample]]
        pos_in_cells = 0.5 * np.ones((len(top_cells), 2))  # Predict cell-centers

        accs = calc_sample_accuracies(pose, top_cells, pos_in_cells, args.top_k, args.threshs)

        for k in args.top_k:
            for t in args.threshs:
                accuracies[k][t].append(accs[k][t])

    for k in args.top_k:
        for t in args.threshs:
            accuracies[k][t] = np.mean(accuracies[k][t])

    return retrievals, accuracies


@torch.no_grad()
def run_fine(model, retrievals, dataloader, args, transform_fine):
    # A batch in this dataset contains max(top_k) times the pose vs. each of the max(top_k) top-cells.

    model.eval()
    dataset_topk = Kitti360TopKDataset(
        dataloader.dataset.all_poses, dataloader.dataset.all_cells, retrievals, transform_fine, args,)

    t0 = time.time()
    # Obtain the matches, offsets and confidences for each pose vs. its top-cells
    offsets = []
    cell_ids = []
    poses_w = []
    
    t0 = time.time()
    pbar = tqdm.tqdm(enumerate(dataset_topk), total = len(dataset_topk))

    # load coarse_model if have
    model_coarse = None
    if args.coarse_path_in_fine is not None:
        model_coarse = CellRetrievalNetwork(
            dataloader.dataset.get_known_classes(),
            COLOR_NAMES_K360,
            args,
        )
        model_coarse_dic = torch.load(args.coarse_path_in_fine, map_location=torch.device("cpu"))
        model_coarse.load_state_dict(model_coarse_dic, strict = False)

    for i_sample, sample in pbar:
        texts = sample["texts"]
        #Calculate similarity matrix in Object-level
        if model_coarse is not None:
            text_enc,text_enc_object_level,text_enc_relation_level = model_coarse.encode_text(texts)
            cell_enc,cell_enc_object_level,cell_enc_relation_level,cell_enc_object_level_mask,cell_enc_relation_level_mask = model_coarse.encode_objects(sample["objects"], sample["object_points"])
            aa=text_enc_object_level
            bb=cell_enc_object_level.transpose(1,2)
            object_level_similarity=torch.matmul(aa,bb)
            output = model(sample["objects"], texts, sample["object_points"],object_level_similarity[:,:,0:args.pad_size],args.confidence_thresh)
            del text_enc,text_enc_object_level,text_enc_relation_level,cell_enc,cell_enc_object_level,cell_enc_relation_level,cell_enc_object_level_mask,cell_enc_relation_level_mask
        else:
            output = model(sample["objects"], texts, sample["object_points"])
        #output = model(sample["objects"], sample["texts"], sample["object_points"])
        offsets.append(output.detach().cpu().numpy())

        cell_ids.append([cell.id for cell in sample["cells"]])
        poses_w.append(sample["poses"][0].pose_w)
    print(f"Ran matching for {len(dataset_topk)} queries in {time.time() - t0:0.2f}.")

    assert len(offsets) == len(retrievals)
    cell_ids = np.array(cell_ids)

    t1 = time.time()
    print("ela:", t1 - t0)

    all_cells_dict = {cell.id: cell for cell in dataloader.dataset.all_cells}

    # Gather the accuracies for each sample
    accuracies_offset = {k: {t: [] for t in args.threshs} for k in args.top_k}
    for i_sample in tqdm.tqdm(range(len(retrievals))):
        pose = dataloader.dataset.all_poses[i_sample]
        top_cells = [all_cells_dict[cell_id] for cell_id in retrievals[i_sample]]
        sample_offsets = offsets[i_sample]

        if not np.all(np.array([cell.id for cell in top_cells]) == cell_ids[i_sample]):
            print()
            print([cell.id for cell in top_cells])
            print(cell_ids[i_sample])

        assert np.all(np.array([cell.id for cell in top_cells]) == cell_ids[i_sample])
        assert np.allclose(pose.pose_w, poses_w[i_sample])

        # Get objects, matches and offsets for each of the top-cells

        pos_in_cells_offsets = []
        for i_cell in range(len(top_cells)):
            # Copy the cell and pad it again, as the fine model might have matched a padding-object
            cell = deepcopy(top_cells[i_cell])
            while len(cell.objects) < args.pad_size:
                cell.objects.append(Object3d.create_padding())

            cell_offsets = sample_offsets[i_cell]
            pos_in_cells_offsets.append(cell_offsets)
        pos_in_cells_offsets = np.array(pos_in_cells_offsets)

        accs_offsets = calc_sample_accuracies(
            pose, top_cells, pos_in_cells_offsets, args.top_k, args.threshs
        )

        for k in args.top_k:
            for t in args.threshs:
                accuracies_offset[k][t].append(accs_offsets[k][t])

    for k in args.top_k:
        for t in args.threshs:
            accuracies_offset[k][t] = np.mean(accuracies_offset[k][t])

    return accuracies_offset


if __name__ == "__main__":
    args = parse_arguments()
    print(str(args).replace(",", "\n"), "\n")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device, torch.cuda.get_device_name(0))

    # Load datasets
    if args.no_pc_augment:
        transform = T.FixedPoints(args.pointnet_numpoints)
    else:
        transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])

    if args.no_pc_augment_fine:
        transform_fine = T.FixedPoints(args.pointnet_numpoints)
    else:
        transform_fine = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])


    if args.use_test_set:
        dataset_retrieval = Kitti360CoarseDatasetMulti(
            args.base_path, SCENE_NAMES_TEST, transform, shuffle_hints=False, flip_poses=False,
        )
    else:
        dataset_retrieval = Kitti360CoarseDatasetMulti(
            args.base_path, SCENE_NAMES_VAL, transform, shuffle_hints=False, flip_poses=False,
        )
    
    dataloader_retrieval = DataLoader(
        dataset_retrieval,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )

    # Load models
    model_coarse_dic = torch.load(args.path_coarse, map_location=torch.device("cpu"))
    model_coarse = CellRetrievalNetwork(
                KNOWN_CLASS,
                COLOR_NAMES_K360,
                args,
            )
    model_coarse.load_state_dict(model_coarse_dic, strict = False)
    model_coarse.to(device)

    if args.path_fine:
        model_fine_dic = torch.load(args.path_fine, map_location=torch.device("cpu"))
        model_fine = CrossMatch(
            KNOWN_CLASS,
            COLOR_NAMES_K360,
            args,
        )
        model_fine.load_state_dict(model_fine_dic, strict = False)
        model_fine.to(device)


    # Run coarse
    retrievals, coarse_accuracies = run_coarse(model_coarse, dataloader_retrieval, args)
    print_accuracies(coarse_accuracies, "Coarse")


    # Run fine
    accuracies_offsets = run_fine(
        model_fine, retrievals, dataloader_retrieval, args, transform_fine
    )
    print_accuracies(accuracies_offsets, "Fine")
