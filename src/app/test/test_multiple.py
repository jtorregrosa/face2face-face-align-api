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
import pytest
from io import BytesIO
from PIL import Image

from .. import application
from . import utils

@pytest.fixture
def client():
    """
    Creates a test client.
    :return: a new created test client
    """
    application.config['TESTING'] = True
    client = application.test_client()

    yield client

def test_single_detection(client):
    """
    Test a simple face detection.
    :param client: test client
    """

    response = utils.upload_file('data/testimage1.jpeg', 64, 'multiple', client)
    json_data = response.get_json()

    assert json_data['inputType'] == 'jpeg', 'Input type must be reported as jpeg'
    assert json_data['targetSize'] == 64, 'Target size must be 64'
    assert json_data['processTime'] is not None, 'Process time must be returned'
    assert json_data['imageCount'] == 4, 'ImageCount must be 4'
    assert len(json_data['data']) == 4, '4 faces must be detected'

def test_detection_sizes(client):
    """
    Test if de sizes are correctly handled.
    :param client: test client
    """
    utils.assert_images_size(utils.upload_file('data/testimage1.jpeg', 37, 'multiple', client).get_json()['data'], 37)
    utils.assert_images_size(utils.upload_file('data/testimage1.jpeg', 256, 'multiple', client).get_json()['data'], 256)

    response = utils.upload_file('data/testimage1.jpeg', 0, 'multiple', client)
    json_data = response.get_json()

    assert response.status_code == 400, 'Status Code must be 400'
    assert json_data['message'] == 'Only sizes between 8 and 1024 are allowed'

    response2 = utils.upload_file('data/testimage1.jpeg', 2048, 'multiple', client)
    json_data2 = response.get_json()

    assert response2.status_code == 400, 'Status Code must be 400'
    assert json_data2['message'] == 'Only sizes between 8 and 1024 are allowed'

def test_no_detection(client):
    """
    Test that if the image doesn't contain faces, no one is detected.
    :param client: test client
    """

    response = utils.upload_file('data/testimage3.jpeg', 64, 'multiple', client)
    json_data = response.get_json()

    assert json_data['inputType'] == 'jpeg', 'Input type must be reported as jpeg'
    assert json_data['targetSize'] == 64, 'Target size must be 64'
    assert json_data['processTime'] is not None, 'Process time must be returned'
    assert json_data['imageCount'] == 0, 'ImageCount must be 4'
    assert len(json_data['data']) == 0, 'No faces must be detected'


def test_wrong_extension(client):
    """
    This test checks that a jpg image with .png extension will be managed as jpg.
    :param client: test client
    """

    response = utils.upload_file('data/testimage2.png', 64, 'multiple', client)
    json_data = response.get_json()

    assert json_data['inputType'] == 'jpeg', 'Input type must be reported as jpeg'
    assert json_data['targetSize'] == 64, 'Target size must be 64'
    assert json_data['processTime'] is not None, 'Process time must be returned'
    assert json_data['imageCount'] == 4, 'ImageCount must be 4'
    assert len(json_data['data']) == 4, '4 faces must be detected'

def test_non_image_upload(client):
    """
    This test checks that a text file must raise an error.
    :param client: test client
    """

    response = utils.upload_file('data/testfile1.txt', 64, 'multiple', client)
    json_data = response.get_json()

    assert response.status_code == 400, 'Status Code must be 400'
    assert json_data['message'] == 'Only png, bmp, jpeg or gif files are allowed', 'An error must be raised'

def test_non_image_upload2(client):
    """
    This test checks that a text file uploaded with .jpg extension must raise an error.
    :param client: test client
    """
    response = utils.upload_file('data/testfile2.jpg', 64, 'multiple', client)
    json_data = response.get_json()

    assert response.status_code == 400, 'Status Code must be 400'
    assert json_data['message'] == 'Only png, bmp, jpeg or gif files are allowed', 'An error must be raised'

def test_binary_upload1(client):
    """
    This test checks that a binary file with a .dat extension must raise an error.
    :param client: test client
    """

    response = utils.upload_file('data/testbin1.dat', 64, 'multiple', client)
    json_data = response.get_json()

    assert response.status_code == 400, 'Status Code must be 400'
    assert json_data['message'] == 'Only png, bmp, jpeg or gif files are allowed', 'An error must be raised'

def test_binary_upload2(client):
    """
    This test checks that a binary file with a jpg extension must raise an error.
    :param client: test client
    """

    response = utils.upload_file('data/testbin2.jpg', 64, 'multiple', client)
    json_data = response.get_json()

    assert response.status_code == 400, 'Status Code must be 400'
    assert json_data['message'] == 'Only png, bmp, jpeg or gif files are allowed', 'An error must be raised'


if __name__ == '__main__':
    unittest.main()