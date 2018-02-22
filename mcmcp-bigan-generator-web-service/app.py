import os

import boto
from flask import (
    Flask,
    jsonify,
    request,
)
import numpy as np

import generate

# Render an image for kicks.
generate.render_image()

app = Flask(__name__)

conn = boto.connect_s3(
    os.environ['aws_access_key_id'],
    os.environ['aws_secret_access_key'],
)
bucket_name = "dlgr-deep"
bucket = conn.get_bucket(bucket_name, validate=False)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/image/', methods=['GET', 'POST'])
def image():
    z = request.values.get('z', None)
    if z:  # Convert to Numpy array.
        z = np.fromstring(z[1:-1], dtype=np.float, sep=',')
    img_path = generate.render_image(z=z)
    url = save_to_s3(img_path)
    response = jsonify(
        status="success",
        data={
            "image_url": url,
        })
    return response


def save_to_s3(img_path):
    (_, filename) = os.path.split(img_path)
    k = boto.s3.key.Key(bucket)
    k.key = filename
    k.set_contents_from_filename(img_path)
    url = "https://{}.s3.amazonaws.com/{}".format(
        bucket_name,
        filename,
    )
    return url


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
