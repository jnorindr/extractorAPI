import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import requests
from urllib.parse import urlparse

from ..src.utils import create_dir
from ..src.utils.logger import console, log


def get_id(dic):
    if type(dic) == list:
        dic = dic[0]

    if type(dic) == dict:
        try:
            return dic["@id"]
        except KeyError:
            try:
                return dic["id"]
            except KeyError as e:
                log(f"No id provided {e}")

    if type(dic) == str:
        return dic

    return None


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
    # return get_img_id(item_img), img_url


def get_img_id(img):
    img_id = get_id(img)
    if ".jpg" in img_id:
        try:
            return img_id.split("/")[-5]
        except IndexError:
            return None
        # return Path(urlparse(img_id).path).parts[-5]
    return img_id.split("/")[-1]


def get_formatted_size(width="", height=""):
    if not width and not height:
        return "full"
    return f"{width or ''},{height or ''}"


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
        img_list = [canvas["images"] for canvas in manifest["sequences"][0]["canvases"]]
        # img_info = [get_canvas_img(img, only_img_url) for imgs in img_list for img in imgs]
        img_info = [get_img_rsrc(img) for imgs in img_list for img in imgs]
    except KeyError:
        try:
            img_list = [
                item
                for items in manifest["items"]
                for item in items["items"][0]["items"]
            ]
            # img_info = [get_item_img(img, only_img_url) for img in img_list]
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


def get_json(url):
    try:
        response = requests.get(url)
        if response.ok:
            return response.json()
        else:
            response.raise_for_status()
    except requests.exceptions.RequestException as e:
        log(e)
        return None


def save_img(img: Image, img_filename, saving_dir, error_msg="Failed to save img"):
    try:
        img.save(saving_dir / img_filename)
    except Exception as e:
        log(f"{error_msg}:\n{e}")


def save_iiif_img(img_rscr, i, work, output_dir, size="full", re_download=False):
    img_name = f"{i:04d}.jpg"

    if os.path.isfile(output_dir / work / img_name) and not re_download:
        # if the img is already downloaded, don't download it again
        return False

    img_url = get_id(img_rscr["service"])
    iiif_url = f"{img_url}/full/{size}/0/default.jpg"

    with requests.get(iiif_url, stream=True) as response:
        response.raw.decode_content = True
        try:
            img = Image.open(response.raw)
        except UnidentifiedImageError:
            if size == "full":
                size = get_reduced_size(img_rscr["width"])
                save_iiif_img(img_rscr, i, work, output_dir, get_formatted_size(size))
                return
            else:
                log(f"Failed to extract images from {img_url}")
                # iiif_log(img_url)
                return

        save_img(img, img_name, output_dir, f"Failed to extract from {img_url}")
    return True


class IIIFDownloader:
    """Download all image resources from a list of manifest urls."""

    def __init__(self, manifest_url, img_dir, width=None, height=None, sleep=0.5):
        self.manifest_url = manifest_url
        self.img_dir = create_dir(img_dir)
        self.size = get_formatted_size(width, height)
        self.sleep = sleep

    def run(self):
        manifest = get_json(self.manifest_url)
        if manifest is not None:
            manifest_id = get_manifest_id(manifest)

            if manifest_id is None:
                console("Unable to retrieve manifest_id")

            console(f"Processing {manifest_id}...")
            if not os.path.exists(self.img_dir / manifest_id):
                img_path = create_dir(
                     self.img_dir / manifest_id
                 )
                i = 1
                for rsrc in get_iiif_resources(manifest):
                    save_iiif_img(rsrc, i, manifest_id, img_path)
                    i += 1
