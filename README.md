# Face2Face - DLib Face Alignment API

![Face2Face Logo](https://raw.githubusercontent.com/jtorregrosa/face2face-dlib-face-alignment-api/master/docs/images/logo.png)

This project is part of a set of projects that make up Face2Face (still in development).

It allows to extract and align faces from an image. Those output images could be used as input for any machine learning 
algorithm that learn how to recognize faces.

This is the first phase in order to build a face recognition pipeline.

![Face2Face Demo 1](https://raw.githubusercontent.com/jtorregrosa/face2face-dlib-face-alignment-api/master/docs/images/demo1.jpg)

The process uses a 68 face landmark shape predictor trained by `C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.`
from https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/ to detect, crop, stretch and align faces in the same way. This allow us to get high quality faces for our algorithms.

 
![Face2Face Demo 2](https://raw.githubusercontent.com/jtorregrosa/face2face-dlib-face-alignment-api/master/docs/images/demo2.jpg)


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and 
testing purposes. 

See deployment for notes on how to deploy the project on a live system.

### Prerequisites

* Python3
* Pip
* Docker
* Make

```
Give examples
```

### Installing

1. Clone this repository in your local system and navigate to root folder.

```bash
$ git clone repository-url
```

2. Navigate to `src` folder and create a virtual environment.

```bash
$ cd src
$ make create-env
```

3. Install required dependencies via prepare make target.

```bash
$ make install
```

4. Run your own copy!

```bash
$ make run
```

5. (OPTIONAL) If you want to run the application with Gunicorn, execute the specific target.

```bash
$ make gunicorn
```

Or if you need custom parameters do the following.

```bash
$ ARGS="--bind 0.0.0.0:8000 --workers 16 --worker-class gthread --threads 16" make gunicorn
```

## Testing

The tests are implemented using PyTest. In order to run them execute:

```bash
$ cd src
$ make test
```

## Usage

When the project is fully deployed, you will get a set of endpoints. Those endpoint will allow to extract faces
present in an image.

### Single align

This endpoint will extract only the largest face in the provided image:

`{base-url}/api/align/single/{size}`

You must provide a path parameter indicating the desired output size. This value must be > 0.

The endpoint accept a content-type of `application/x-www-form-urlencoded` with a parameter `file` holding your image.

An example of the response:

```json
{
    "processTime": 0.49076399999999865, 
    "targetSize": 12, 
    "inputType": "png", 
    "data": "base64-image-1"
}
```

### Multiple align

This endpoint will extract all the faces present in the provided image.

`{base-url}/api/align/single/{size}`

You must provide a path parameter indicating the desired output size. This value must be > 0.

The endpoint accept a content-type of `application/x-www-form-urlencoded` with a parameter `file` holding your image.

An example of the response:

```json
{
    "processTime": 0.49076399999999865, 
    "targetSize": 12, 
    "inputType": "png", 
    "imageCount": 1, 
    "data": [
        "base64-image-1",
        "base64-image-2",
        "base64-image-3"
    ]
}
```

## Built With

* [Flask](http://flask.pocoo.org/) - The web framework used
* [Flask-RESTPlus](https://flask-restplus.readthedocs.io/en/stable/) - Flask REST Extensions
* [Swagger](https://swagger.io/) - API Documentation
* [DLib](http://dlib.net/) - C++ toolkit containing machine learning algorithms

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Jorge Torregrosa** - *Main Developer* - [jtorregrosa](https://github.com/jtorregrosa)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

### Project License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

### Additional Licenses
#### Landmarks 68 Dlib model 
*The license for this dataset excludes commercial use*.  So you should contact a lawyer or talk to Imperial College 
London to find out if it's OK for you to use this model in a commercial product.

https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/

```
C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. 
300 faces In-the-wild challenge: Database and results. 
Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
```

You may need a self-trained model to use this project for commercial purposes.

## Acknowledgments

* **C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic** - For their amazing job in face annotation tasks
* **Greg Obinna** - For his post about Flask  fro production (https://medium.freecodecamp.org/structuring-a-flask-restplus-web-service-for-production-builds-c2ec676de563)
* **Cole Murray** - For his post about Facial Recognition using TensorFlow (https://hackernoon.com/building-a-facial-recognition-pipeline-with-deep-learning-in-tensorflow-66e7645015b8)
* **David Sanberg** - For his research and work on FaceNet (https://github.com/davidsandberg/facenet)