"""
Copyright 2019 Jorge Torregrosa

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import base64
from io import BytesIO
from PIL import Image

def assert_images_size(images_b64, size):
    """
    Assert that the provided base64 images size is correct.
    :param images_b64: base64 string array
    :param size: expected size
    """
    for image in images_b64:
        assert_image_size(image, size)

def assert_image_size(image_b64, size):
    """
    Assert that the provided base64 image size is correct.
    :param image_b64: base64 string
    :param size: expected size
    """
    decoded_image = base64.b64decode(image_b64)
    f = BytesIO()
    f.write(decoded_image)
    f.seek(0)
    Image.open(f)

    with Image.open(f) as image:
        assert image.size[1] == size, "Size must be " + size

def upload_file(filepath, size, mode, client):
    """
    Uploads a file and return its response.
    :param filepath: file path
    :param size: output image size
    :param mode: detection mode (single|multiple)
    :param client: test client
    :return: api response
    """
    with open(os.path.join(os.path.dirname(__file__), filepath), 'rb') as file:
        rv = client.post('/api/align/' + mode + '/' + str(size), data=dict(
            file=file
        ))

        return rv