import os
import time
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import requests
from urllib.parse import urlparse

from utils import create_dir, get_json, check_if_dir_exists
from utils.logger import console, log
from utils.paths import IMG_PATH


def get_img_rsrc(iiif_img):
    try:
        img_rscr = iiif_img["resource"]
    except KeyError:
        try:
            img_rscr = iiif_img["body"]
        except KeyError:
            return None
    return img_rscr


def get_canvas_img(canvas_img, only_img_url=False):
    img_url = get_id(canvas_img["resource"]["service"])
    if only_img_url:
        return img_url
    return get_img_id(canvas_img["resource"]), img_url


def get_item_img(item_img, only_img_url=False):
    img_url = get_id(item_img["body"]["service"][0])
    if only_img_url:
        return img_url
    return get_img_id(item_img), img_url


def get_img_id(img):
    img_id = get_id(img)
    if ".jpg" in img_id:
        try:
            return img_id.split("/")[-5]
        except IndexError:
            return None
        # return Path(urlparse(img_id).path).parts[-5]
    return img_id.split("/")[-1]


def get_manifest_id(manifest):
    manifest_id = get_id(manifest)
    if "manifest" in manifest_id:
        try:
            return Path(urlparse(get_id(manifest)).path).parent.name
        except Exception:
            return None
    return manifest_id.split("/")[-1]


def get_iiif_resources(manifest, only_img_url=False):
    try:
        # Usually images URL are contained in the "canvases" field
        img_list = [canvas["images"] for canvas in manifest["sequences"][0]["canvases"]]
        img_info = [get_img_rsrc(img) for imgs in img_list for img in imgs]
    except KeyError:
        # But sometimes in the "items" field
        try:
            img_list = [
                item
                for items in manifest["items"]
                for item in items["items"][0]["items"]
            ]
            img_info = [get_img_rsrc(img) for img in img_list]
        except KeyError as e:
            log(f"Unable to retrieve resources from manifest {manifest}\n{e}")
            return []

    return img_info


def get_reduced_size(size, min_size=1500):
    size = int(size)
    if size < min_size:
        return ""
    if size > min_size * 2:
        return str(int(size / 2))
    return str(min_size)


def save_img(img: Image, img_filename, error_msg="Failed to save img"):
    try:
        img.save(IMG_PATH / img_filename)
    except Exception as e:
        log(f"{error_msg}:\n{e}")


def get_id(dic):
    if isinstance(dic, list):
        dic = dic[0]

    if isinstance(dic, dict):
        try:
            return dic["@id"]
        except KeyError:
            try:
                return dic["id"]
            except KeyError as e:
                log(f"No id provided {e}")

    if isinstance(dic, str):
        return dic

    return None


class IIIFDownloader:
    """Download all image resources from a list of manifest urls."""

    def __init__(self, manifest_url, img_dir, width=None, height=None, sleep=0.5, max_dim=2000):
        self.manifest_url = manifest_url
        self.img_dir = create_dir(img_dir)
        self.size = self.get_formatted_size(width, height)
        self.sleep = sleep
        self.max_dim = max_dim  # Can be set to None
        self.manifest_dir = self.manifest_url.replace("/", "").replace(".", "")

    def run(self):
        manifest = get_json(self.manifest_url)
        if manifest is not None:

            console(f"Processing {self.manifest_url}...")
            # TODO: here set img dir as paths.py variable + use create_if_not_dir
            # TODO: download img only if not exists
            if not check_if_dir_exists(self.img_dir / self.manifest_dir):
                i = 1
                for rsrc in get_iiif_resources(manifest):
                    self.save_iiif_img(rsrc, i)
                    i += 1

    #      def extract_images_from_iiif_manifest(self, manifest_url, work):
    #         """
    #         Extract all images from an IIIF manifest
    #         """
    #         manifest = get_json(manifest_url)
    #         if manifest is not None:
    #             i = 1
    #             for img_rscr in get_iiif_resources(manifest, True):
    #                 is_downloaded = self.save_iiif_img(img_rscr, i, work)
    #                 i += 1
    #                 if is_downloaded:
    #                     time.sleep(5 if "gallica" in manifest_url else 0.25)

    def get_formatted_size(self, width="", height=""):
        if not width and not height:
            if self.max_dim is not None:
                return f",{self.max_dim}"
            return "full"

        if self.max_dim is not None and int(width) > self.max_dim:
            width = f"{self.max_dim}"
        if self.max_dim is not None and int(height) > self.max_dim:
            height = f"{self.max_dim}"

        return f"{width or ''},{height or ''}"

    def save_iiif_img(self, img_rscr, i, size="full", re_download=False):
        img_name = f"{i:04d}.jpg"  # TODO name eida img as ms-blablab_0001.jpg

        # img_name = f"{manifest_dir}_{i:04d}.jpg"
        #
        #     if os.path.isfile(BASE_DIR / IMG_PATH / img_name) and not re_download:
        #         # if the img is already downloaded, don't download it again
        #         return False
        #
        #     img_url = get_id(img_rscr["service"])
        #     iiif_url = f"{img_url}/full/{size}/0/default.jpg"
        #
        #     with requests.get(iiif_url, stream=True) as response:
        #         response.raw.decode_content = True
        #         try:
        #             img = Image.open(response.raw)
        #         except UnidentifiedImageError:
        #             if size == "full":
        #                 size = get_reduced_size(img_rscr["width"])
        #                 save_iiif_img(img_rscr, i, manifest_dir, get_formatted_size(size))
        #                 return
        #             else:
        #                 log(f"[save_iiif_img] Failed to extract images from {img_url}")
        #                 return
        #
        #         save_img(img, img_name, f"Failed to extract from {img_url}")
        #     return True

        if os.path.isfile(IMG_PATH / self.manifest_dir / img_name) and not re_download:
            # if the img is already downloaded, don't download it again
            return False

        img_url = get_id(img_rscr["service"])
        iiif_url = f"{img_url}/full/{size}/0/default.jpg"
        # TODO: https://gallica.bnf.fr/iiif/ark:/12148/bpt6k98074966/f97/full/pct:60/0/native.jpg

        with requests.get(iiif_url, stream=True) as response:
            response.raw.decode_content = True
            try:
                img = Image.open(response.raw)
            except UnidentifiedImageError:
                if size == "full":
                    size = get_reduced_size(img_rscr["width"])
                    self.save_iiif_img(img_rscr, i, self.get_formatted_size(size))
                    return
                else:
                    log(f"Failed to extract images from {img_url}")
                    return

            save_img(img, img_name, f"Failed to extract from {img_url}")
        return True
