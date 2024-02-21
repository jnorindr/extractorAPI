import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.metrics import pairwise
from tqdm import tqdm

from app.similarity.const import SCORES_PATH, SEG_STRIDE, MAX_SIZE, SEG_TOPK, COS_TOPK, FEAT_NET, FEAT_SET, FEAT_LAYER
from app.similarity.dataset import IllusDataset
from app.similarity.features import extract_features
from app.similarity.segswap import load_backbone, load_encoder, resize, compute_score
from app.similarity.utils import get_doc_dirs, get_device, hash_pair, get_model_path, filename
from app.utils.logger import console


def get_cos_pair(doc_pair, img1, img2):
    doc1, doc2 = doc_pair
    img1_dir, img2_dir = img1.split("/")[-2], img2.split("/")[-2]

    if img1_dir == doc1 and img2_dir == doc2:
        return img1, img2
    elif img1_dir == doc2 and img2_dir == doc1:
        return img2, img1
    return None, None


def cosine_similarity(doc_pair):
    img_dataset = IllusDataset(
        img_dirs=get_doc_dirs(doc_pair),
        transform=['resize', 'normalize'],
        device=get_device()
    )

    device = get_device()
    data_loader = DataLoader(img_dataset, batch_size=128, shuffle=False)

    # TODO Extract features in previous step
    features = extract_features(data_loader, device, FEAT_LAYER, FEAT_SET, FEAT_NET, hash_pair(doc_pair)).cpu().numpy()
    if not len(features):
        console("Error when extracting features", color="red")
        raise ValueError

    sim = pairwise.cosine_distances(features)

    img_paths, _ = img_dataset.get_image_paths()
    sim_pairs = []

    tr_ = transforms.Resize((224, 224))
    i = 0
    for query_img in img_paths:
        # TODO maybe resize directly inside save_img()
        img = cv2.imread(query_img)
        img = torch.from_numpy(img).permute(2, 0, 1)
        tr_img = tr_(img).permute(1, 2, 0).numpy()
        cv2.imwrite(query_img, tr_img)

        res = [1.0 if query_img == img_p else sim_score for img_p, sim_score in zip(img_paths, sim[i])]
        sorted_indices_by_res = np.argsort(res)[:-1][: COS_TOPK] # [:-1] to remove the query image from its results
        sim_imgs = [img_paths[idx] for idx in sorted_indices_by_res]
        # sim_pairs.extend([(query_img, sim_img) for sim_img in sim_imgs])
        sim_pairs.append([get_cos_pair(doc_pair, query_img, sim_img) for sim_img in sim_imgs])
        i += 1

    np_pairs = np.array(sim_pairs)
    return np_pairs

def segswap_similarity(cos_pairs, output_file=None):
    param = torch.load(get_model_path('hard_mining_neg5'))
    backbone = load_backbone(param)
    encoder = load_encoder(param)

    feat_size = MAX_SIZE // SEG_STRIDE
    mask = np.ones((feat_size, feat_size), dtype=bool)
    y_grid, x_grid = np.where(mask)

    scores_npy = np.empty((0, 3), dtype=object)

    norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transformINet = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)]
    )

    for p in tqdm(range(cos_pairs.shape[0])):
        # img_pairs = [(img1doc1, img1doc2), (img1doc1, img2doc2), ...] # pairs for img1doc1
        img_pairs = cos_pairs[p]
        if img_pairs[0, 0] is None:
            break
        q_img = filename(img_pairs[0, 0])
        # q_img = img_pairs[0, 0]

        qt_img = resize(Image.open(img_pairs[0, 0]).convert("RGB"))
        q_tensor = transformINet(qt_img).cuda()
        sim_imgs = img_pairs[:, 1]

        tensor1 = []
        tensor2 = []
        for s_img in sim_imgs[:SEG_TOPK]:
            if s_img is None:
                break
            st_img = resize(Image.open(s_img).convert("RGB"))
            tensor1.append(q_tensor) # NOTE: maybe not necessary to duplicate same img tensor
            tensor2.append(transformINet(st_img).cuda())

        score = compute_score(torch.stack(tensor1), torch.stack(tensor2), backbone, encoder, y_grid, x_grid)

        for i in range(len(score)):
            pair_score = np.array([[round(score[i], 5), q_img, filename(sim_imgs[i])]])
            # pair_score = np.array([[round(score[i], 5), q_img, sim_imgs[i]]])
            scores_npy = np.vstack([scores_npy, pair_score])

    if output_file:
        try:
            np.save(SCORES_PATH / f"{output_file}.npy", scores_npy)
        except Exception as e:
            console(f"Failed to save {output_file}.npy", error=e)

    return scores_npy


def compute_seg_pairs(doc_pair, hashed_pair):
    console(f"COMPUTING SIMILARITY FOR {doc_pair} 🖇️")
    try:
        console(f"Computing cosine scores for {doc_pair} 🖇️", color="cyan")
        cos_pairs = cosine_similarity(doc_pair)
        # Reshape to group pairs for same query img
        # cos_pairs = cos_pairs.reshape(-1, COS_TOPK, cos_pairs.shape[1])
    except Exception as e:
        console(f"Error when computing cosine similarity", error=e)
        return np.empty(0)

    try:
        console(f"Computing segswap scores for {doc_pair} 🖇️", color="cyan")
        segswap_similarity(cos_pairs, output_file=hashed_pair)
    except Exception as e:
        console(f"Error when computing segswap scores", error=e)
        return False
    return True
