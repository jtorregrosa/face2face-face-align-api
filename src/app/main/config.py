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

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
    """
    Base configuration for all environments.
    """
    SECRET_KEY = os.getenv('SECRET_KEY', 'my_secret_key')
    DEBUG = False
    JSON_SORT_KEYS = True
    JSONIFY_MIMETYPE = 'application/json'


class DevelopmentConfig(Config):
    """
    Development time specific configuration.
    """
    DEBUG = True
    JSONIFY_PRETTYPRINT_REGULAR = True


class TestingConfig(Config):
    """
    Testing time specific configuration.
    """
    DEBUG = True
    TESTING = True


class ProductionConfig(Config):
    """
    Production time specific configuration.
    """
    DEBUG = False
    SECRET_KEY = os.getenv('SECRET_KEY')


config_by_name = dict(
    dev=DevelopmentConfig,
    test=TestingConfig,
    prod=ProductionConfig
)

key = Config.SECRET_KEY
