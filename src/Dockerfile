#
# Copyright 2019 Jorge Torregrosa
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

FROM python:3.7.2-stretch

LABEL maintainer="jtorregrosalloret@gmail.com"

ENV APP_ENV='prod'
ENV SECRET_KEY='yoursupersecretkey'
ENV NUM_WORKERS=16
ENV WORKER_CLASS='gthread'
ENV NUM_THREADS_PER_WORKER=16

# Expose main API port
EXPOSE 8000

# Install required system packages
RUN apt-get update -y && \
    apt-get install -y apt-utils && \
    apt-get install -y python-pip python-dev build-essential cmake libopenblas-dev liblapack-dev --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Copy first requirements to use docker cache with dependencies
COPY requirements.txt ./app/requirements.txt

# Set current working directory to application folder
WORKDIR /app

# Install required application packages
RUN python3 -m venv venv && \
    . ./venv/bin/activate && \
    pip install --disable-pip-version-check --upgrade setuptools wheel && \
    pip install --disable-pip-version-check -r requirements.txt

# Copy application data
COPY . /app

RUN chmod +x ./entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]