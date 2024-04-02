import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.metrics import pairwise
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

from app.similarity.const import (
    SCORES_PATH,
    SEG_STRIDE,
    MAX_SIZE,
    SEG_TOPK,
    COS_TOPK,
    FEAT_NET,
    FEAT_SET,
    FEAT_LAYER,
    DOC_PATH
)
from app.similarity.dataset import IllusDataset
from app.similarity.features import extract_features
from app.similarity.segswap import load_backbone, load_encoder, resize, compute_score
from app.similarity.utils import get_doc_dirs, get_device, hash_pair, get_model_path, filename
from app.utils.logger import console


def get_cos_pair(doc_pair, img1, img2):
    """
    Allows to order the two images according to the doc pair order
    in order to have all the scores for a same document on the same score "column"
    """
    doc1, doc2 = doc_pair
    img1_dir, img2_dir = img1.split("/")[-2], img2.split("/")[-2]

    if img1_dir == doc1 and img2_dir == doc2:
        return img1, img2
    elif img1_dir == doc2 and img2_dir == doc1:
        return img2, img1

    # None values allows to filter out the unwanted pairs
    return None, None


def get_doc_feat(doc_id):
    img_dataset = IllusDataset(
        img_dirs=DOC_PATH / doc_id,
        transform=['resize', 'normalize'],
        device=get_device()
    )

    device = get_device()
    data_loader = DataLoader(img_dataset, batch_size=128, shuffle=False)

    features = extract_features(data_loader, device, FEAT_LAYER, FEAT_SET, FEAT_NET, doc_id).cpu().numpy()
    if not len(features) or not features or type(features) is not np.ndarray:
        console("[get_doc_feat] Error when extracting features: feature tensor is empty.", color="red")
        # print("[get_doc_feat] Error when extracting features: feature tensor is empty.")
        raise ValueError
    return features, img_dataset.get_image_paths()


def doc_sim_pairs(sim_scores, query_doc, sim_doc, is_doc_1=True):
    sim_pairs = []
    tr_ = transforms.Resize((224, 224))
    for i, query_img in enumerate(query_doc):
        # # TODO is it necessary to perform those operations?
        img = cv2.imread(query_img)
        img = torch.from_numpy(img).permute(2, 0, 1)
        tr_img = tr_(img).permute(1, 2, 0).numpy()
        cv2.imwrite(query_img, tr_img)

        query_scores = sim_scores[:][i] if is_doc_1 else sim_scores[i, :]
        # set 0 as similarity score for the query image to itself
        query_scores = [0.0 if query_img == sim_img else img_score for sim_img, img_score in zip(query_doc, query_scores)]

        top_indices = np.argsort(query_scores)[-COS_TOPK:][::-1]
        sim_pairs.append([
            (query_img, sim_doc[j]) if is_doc_1 else (sim_doc[j], query_img)
            for j in top_indices
        ])

    return sim_pairs


def cosine_similarity(doc_pair):
    doc1_feat, doc1_imgs = get_doc_feat(doc_pair[0])
    doc2_feat, doc2_imgs = get_doc_feat(doc_pair[1])

    sim = cos_sim(doc1_feat, doc2_feat)  # sim has shape (n_img_doc1, n_img_doc2)
    sim_pairs = doc_sim_pairs(sim, doc1_imgs, doc2_imgs)
    np_pairs = np.array(sim_pairs)

    """
    # If we don't assume that all the best matching images of doc2 in doc1
    # are already contained in best matching images of doc1 in doc2
    sim_pairs += doc_sim_pairs(sim, doc2_imgs, doc1_imgs, False)

    # Remove duplicates pairs with list(set())
    np_pairs = np.array(list(set(sim_pairs)))
    """

    # np_pairs = [[(img1doc1, img1doc2), (img1doc1, img2doc2), ...]  # best matching images for img1doc1
    #             [(img2doc1, img4doc2), (img2doc1, img8doc2), ...]] # best matching images for img2doc1
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

    """
    first_imgs = cos_pairs[:, 0]
    # get a unique list of all the first img in pairs of cosine similar images
    query_imgs = np.unique(first_imgs)
    for q_img in query_imgs:
        img_pairs = cos_pairs(np.char.startswith(first_imgs, q_img))
    """

    # img_pairs = [(img1doc1, img1doc2), (img1doc1, img2doc2), ...] # pairs for img1doc1
    for img_pairs in cos_pairs:
        q_img = filename(img_pairs[0, 0])

        qt_img = resize(Image.open(img_pairs[0, 0]).convert("RGB"))
        q_tensor = transformINet(qt_img).cuda()
        sim_imgs = img_pairs[:, 1]

        # tensor1 = []
        tensor1 = [q_tensor] * len(sim_imgs)
        tensor2 = []
        for s_img in sim_imgs:  # sim_imgs[:SEG_TOPK]
            st_img = resize(Image.open(s_img).convert("RGB"))
            # tensor1.append(q_tensor)  # NOTE: maybe not necessary to duplicate same img tensor
            tensor2.append(transformINet(st_img).cuda())

        score = compute_score(torch.stack(tensor1), torch.stack(tensor2), backbone, encoder, y_grid, x_grid)

        for i in range(len(score)):
            s_img = sim_imgs[i]
            pair_score = np.array([[round(score[i], 5), q_img, filename(s_img)]])
            scores_npy = np.vstack([scores_npy, pair_score])

    if output_file:
        try:
            np.save(SCORES_PATH / f"{output_file}.npy", scores_npy)
        except Exception as e:
            console(f"[segswap_similarity] Failed to save {output_file}.npy", error=e)
            # print(f"[segswap_similarity] Failed to save {output_file}.npy: {e}")

    # scores_npy = [(score, img1doc1, img1doc2), # each cosine pair of image is given a score
    #               (score, img1doc1, img2doc2),
    #               (score, img2doc1, img4doc2),
    #               (score, img2doc1, img8doc2),
    #                ... ]
    return scores_npy


def compute_seg_pairs(doc_pair, hashed_pair):
    console(f"COMPUTING SIMILARITY FOR {doc_pair} 🖇️")
    # print(f"COMPUTING SIMILARITY FOR {doc_pair} 🖇️")
    try:
        console(f"Computing cosine scores for {doc_pair} 🖇️", color="cyan")
        # print(f"Computing cosine scores for {doc_pair} 🖇️")
        cos_pairs = cosine_similarity(doc_pair)
    except Exception as e:
        console(f"Error when computing cosine similarity", error=e)
        # print(f"Error when computing cosine similarity: {e}")
        return False

    try:
        console(f"Computing segswap scores for {doc_pair} 🖇️", color="cyan")
        # print(f"Computing segswap scores for {doc_pair} 🖇️")
        segswap_similarity(cos_pairs, output_file=hashed_pair)
    except Exception as e:
        # console(f"Error when computing segswap scores", error=e)
        print(f"Error when computing segswap scores: {e}")
        return False
    return True
