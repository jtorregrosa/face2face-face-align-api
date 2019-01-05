.PHONY: clean system-packages python-packages install tests run all gunicorn create-env activate-env deactivate-env

clean:
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.log' -delete

create-env:
	python3 -m venv venv

system-packages:
	sudo apt install python-pip libopenblas-dev liblapack-dev -y

python-packages:
	pip install -r requirements.txt

install: system-packages python-packages

run:
	export FLASK_APP=app
	flask run

gunicorn:
	gunicorn 'app' ${ARGS}

test:
	pytest

all: clean install tests run