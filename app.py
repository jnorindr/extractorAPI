import subprocess

from flask import Flask, request


app = Flask(__name__)


@app.route('/run_detect', methods=['POST'])
def run_detect():
    # Model to use for inference
    model = 'yolov5/best.pt'
    # Manifest list file : to be modified to directly be the IIIF manifest URI sent by app
    file = request.files['file']
    # Getting file name to launch script in command line
    file_name = file.filename
    # Calling inference script as a subprocess for it to run in the background
    subprocess.call(['python', 'yolov5/detect_vhs.py', '-f', file_name, '-m', model])

    return 'Detecting diagrams....'


if __name__ == '__main__':
    app.run()
