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

import cv2
import os
import logging
import numpy as np

from ..services.dlibservice import DlibService


class FaceAlignService:
    logger = logging.getLogger(__name__)
    align_dlib = DlibService(os.path.join(os.path.dirname(__file__), '../data/shape_predictor_68_face_landmarks.dat'))

    @staticmethod
    def __decode_image(file):
        """
        Decode an image from given file.
        :param file: source file.
        :return: a decoded image.
        """
        # Read file.
        file_content = file.read()

        # Create numpy matrix from file content.
        np_img = np.frombuffer(file_content, np.uint8)

        # Decode image.
        decoded_img = cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)

        # Convert image from BGR to RGB
        final_img = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)

        return final_img

    def extract_largest_face(self, file, crop_dim):
        """
        Extracts the largest face present in given image file.
        :param file: source file
        :param crop_dim: output crop dimensions
        :return: the largest face image
        """
        # Read file content and decode target image.
        image = FaceAlignService.__decode_image(file)

        if image is None:
            raise IOError('Error reading image')

        # Get largest face BB.
        bb = self.align_dlib.get_largest_face_bb(image)

        # Extract faces.
        faces = self.__extract_portions(image, [bb], crop_dim)

        # Return the first and the only face in the list.
        return faces[0] if len(faces) > 0 else None

    def extract_all_faces(self, file, crop_dim):
        """
        Extracts all faces present in given image file.
        :param file: source file
        :param crop_dim: output crop dimensions
        :return: an image collection of all faces
        """
        # Read file content and decode target image.
        image = FaceAlignService.__decode_image(file)

        # Get largest face BB.
        bbs = self.align_dlib.get_all_face_bbs(image)

        if image is None:
            raise IOError('Error buffering image')

        # Extract faces.
        faces = self.__extract_portions(image, bbs, crop_dim)

        # Return all detected faces.
        return faces

    def __extract_portions(self, image, bbs, crop_dim):
        """
        Extracts face image portions from given image and bounding boxes.
        :param image: source image
        :param bbs: target bounding boxes
        :param crop_dim: output crop dimensions
        :return: a face image collection of all portions
        """
        aligned_images = []

        for bb in bbs:
            aligned = self.align_dlib.align(crop_dim, image, bb, landmark_indices=DlibService.INNER_EYES_AND_BOTTOM_LIP)
            if aligned is not None:
                aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
                aligned_images.append(aligned)

        return aligned_images
