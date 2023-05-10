from flask import Flask, request, jsonify
import os
import sys
from pathlib import Path
from os.path import exists

from yolov5.utils.general import (LOGGER, cv2)
from yolov5.iiif_downloader.src.iiif_downloader import IIIFDownloader
from yolov5.detect_vhs import run_vhs


app = Flask(__name__)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


@app.route('/run_detect', methods=['POST'])
def run_detect():
    manifest_url = request.form['manifest_url']
    model = ROOT / 'yolov5' / 'best.pt'
    rerun = 'False'  # Re-detect already detected images?  # TODO

    img_dir = 'img'
    downloader = IIIFDownloader(manifest_url, img_dir=img_dir)
    downloader.run()

    output_dir = 'annotation'
    if not exists(ROOT / output_dir):
        os.mkdir(ROOT / output_dir)

    for wit_dir in os.listdir(f'{img_dir}'):
        wit_id = wit_dir.replace("ms-", "")
        if exists(f"{output_dir}/{wit_id}.txt"):
            continue
        LOGGER.info(f"\n\n\x1b[38;5;226m\033[1mDETECTING VISUAL ELEMENTS FOR {wit_id} 🕵️\x1b[0m\n\n")
        wit_path = f'{ROOT}/{img_dir}/{wit_dir}'
        for i, img in enumerate(sorted(os.listdir(wit_path)), 1):
            # if exists(f"{output_dir}/{wit_id}.txt"):
            #     # if annotation are generated again, empty annotation file
            #     open(f"{output_dir}/{wit_id}.txt", 'w').close()

            LOGGER.info(f"\n\x1b[38;5;226m===> Processing {img} 🔍\x1b[0m\n")
            run_vhs(weights=model, source=Path(f"{wit_path}/{img}"), anno_file=f"{output_dir}/{wit_id}.txt", img_nb=i)

    return jsonify({'success': True})


if __name__ == '__main__':
    app.run()
