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
from flask import Flask

from .config import config_by_name
from flask_restplus import Api
from flask_cors import CORS
from flask import Blueprint

from .controller.singlealigncontroller import api as singlealign_ns
from .controller.multiplealigncontroller import api as multiplealign_ns

blueprint = Blueprint('api', __name__)

api = Api(blueprint,
          title='Face2Face Face Alignment API',
          version='1.0',
          description='Face2Face DLib face alignment'
          )

api.add_namespace(singlealign_ns, path='/api/align/single')
api.add_namespace(multiplealign_ns, path='/api/align/multiple')

def create_app(config_name):
    """
    Create an app instance with a specific config for given name.
    :param config_name: configuration name.
    :return: An app instance with a specific config for given name.
    """
    app = Flask(__name__)
    app.config.from_object(config_by_name[config_name])
    app.register_blueprint(blueprint)
    app.app_context().push()
    app.url_map.strict_slashes = False
    CORS(app)

    return app
