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


from flask_restplus import Namespace, fields


class SingleDetectionImageDto:
    """
    Dto emitted from single align endpoints.
    """
    api = Namespace('Single Align', description='Single face align operations')
    single_response = api.model('SingleFaceAlignResponse', {
        'processTime': fields.Float(min=0),
        'targetSize': fields.Integer(min=0),
        'inputType': fields.String(min=0),
        'data': fields.String(min=0)
    })


class MultipleDetectionImageDto:
    """
    Dto emitted from multiple align endpoints.
    """
    api = Namespace('Multiple Align', description='Multiple face align operations')
    multiple_response = api.model('MultipleFaceAlignResponse', {
        'processTime': fields.Float(min=0),
        'targetSize': fields.Integer(min=0),
        'inputType': fields.String(min=0),
        'imageCount': fields.Integer(min=0),
        'data': fields.List(fields.String)
    })
