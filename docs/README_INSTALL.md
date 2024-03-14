# Automated Discovery Tool

NOTE: this README is deprecated

We're pleased to introduce Automated Discovery Tool, a software for assisted
automated discovery and exploration of complex systems.

## Installation

### Docker installation

Our software leverages docker to avoid you installing multiple packages. To
install docker:

1. Install Docker: [`LINK`](https://docs.docker.com/engine/install/)
2. Install Docker-compose: [`LINK`](https://docs.docker.com/compose/install/)

## Using Automated Discovery Tool

Our software consists of two primary parts:

- a Python library called [`adtool`](libs/adtool) where you can
  use/download/implement systems to explore as well as exploration algorithms.
- a web application allowing you to perform assisted automated discovery
  experiments using a user-friendly interface.

The `adtool` library can be used independently of the rest of Automated
Discovery Tool, for example to implement some bespoke remote computing
solutions, but its main purpose is to present a standardized interface for
managing experimental data and configurational metadata which is modified and
displayed by the GUI.

### Launching the software

1. Clone the software into a folder on your local computer, using e.g.,

```bash
git clone https://github.com/flowersteam/adtool.git --branch prod
```

2. Create a file `services/prod/.env` with the environment variables appropriate
   for your setup. A sample is provided at `services/prod/sample.env`
3. Run the following command: `start_app.py --mode prod`
4. Navigate to `http://localhost:4200` in a browser

### Using the web app

[User documentation](#user-documentation)

### Modifying the lib

Before being able to modify the lib, you must install it. For this, please refer
to [this](#autodisc-lib). Our lib relies on classes we call `Modules`. Every
system or explorer is a `Module`. Here is a tutorial on how to add a new module
to the lib (and then be able to use it in an experiment launched from the web
app): [Module adding tutorial](#add-a-new-module-to-the-libs).

We also provide callbacks to experiments (e.g. to save discoveries). If you want
to add a new one, you can follow this
[tutorial](#add-a-new-callback-to-the-libs).

### Launching experiments on a remote server

By default, experiments will be launched locally. However, some large
experiments can require large compute. We thus allow users to provide a
configuration to reach a remote server: [Adding remote
server](#add-a-new-remote-server).

---

## Contributing

Please follow these instructions when contributing on the project.

### Installation

The software is implemented in various microservices

- `adtool` library : a pure Python API for implementing custom experiments \
  and search algorithms to use in the software
- AutoDiscServer : simple service which performs the operations specified by
  functionality in `adtool` library.
- FrontEndApp : web GUI implemented in Angular for interacting with the software
- AppDB & ExpeDB : databases which manage the metadata of experiments (linked
  with the FrontEnd) and resulting experimental data, respectively
- JupyterLab : Jupyter instance allowing interaction with experimental data

#### `adtool` library

1. If you do not already have it, please install
   [Conda](https://www.anaconda.com/)
2. Create _autoDiscTool_ conda environment: `conda env create --name
autoDiscTool python=3.11 `
3. Activate _autoDiscTool_ conda environment: `conda activate autoDiscTool`
4. Go to package: `cd libs/adtool`
5. Install package: `pip install -e .`

#### AutoDiscServer

1. Install flask: `pip install -r
services/AutoDiscServer/flask_app/requirements.txt`

### FrontEndApp

1. Install Angular: [`LINK`](https://angular.io/guide/setup-local)
2. Enter the front-end app folder: `cd services/FrontEndApp/angular_app/`.
3. Install required packages: `npm install`

### AppDB & ExpeDB

1. Install Docker: [`LINK`](https://docs.docker.com/engine/install/)
2. Install Docker-compose: [`LINK`](https://docs.docker.com/compose/install/)
3. Go to the service folder: `cd services`.
4. Generate containers `sudo docker-compose -f services/docker-compose.yml
create`
5. Install the Expe DB REST API requirements: `pip install -r
ExpeDB/API/flask_app/requirements.txt`

#### JupyterLab

1. Install jupyter: `pip install -r JupyterLab/requirements.txt`

### Testing the adtool library alone

1. Edit the `libs/tests/AutoDiscExperiment.py` file to configure the experiment
2. Launch the experiment: `python libs/tests/AutoDiscExperiment.py`

### Commits

Please attach every commit to an issue or a merge request. For issues, add #ID
at the beginning of your commit message (with ID the id of the issue).

### Starting the project

#### Prod mode

Launch: `sudo ./start_app.sh`

#### Debug mode

Go to the `services` folder: `cd services`.

##### AutoDiscServer

Launch the flask server: `python -m AutoDiscServer.app`.

##### App DB

Start services: `sudo docker-compose up app-db-api` Add `-d` option for daemon.

##### Expe DB

Start service: `sudo docker-compose up expe-db` Add `-d` option for daemon.
Launch flask server for the REST API: `python ExpeDB/app.py`

##### Front-end app

Enter the front-end app folder: `cd FrontEndApp`. Start the angular app: `ng
serve`.

##### Jupyter Lab

Enter Jupyter Lab's folder: `cd ../JupyterLab` Give the appropriate rights to
the notebook folder: `chmod 777 -R Notebooks/` Start the jupyter lab on port
8887: `export PYTHONPATH=$(pwd)/../../libs/adtool_db; jupyter lab Notebooks/
--config Config/jupyter_notebook_config.py`

### Using monitoring services

We added monitoring applications for the Docker services. You can start then by
going in `services`: `cd services` and launching the following command:

```
sudo docker-compose --profile monitoring up
```

_Warning_: This command will start all the docker services or attach the already
launched ones otherwise.

#### Portainer

Portainer is a web application allowing to monitor Docker environments. You can
access it at: [https://localhost:9443](https://localhost:9443).

At first startup you'll be asked to create a password for your admin account.
Then, you should see your local environment with all the running containers. You
can for instance easily see the logs of each container.

#### PgAdmin

PgAdmin is a lightweight web application allowing to visually interact with a
PostgreSQL database. You can use it to monitor and modify our AppDB (which
should be connected by default). For this, go to
[http://localhost:5050](http://localhost:5050). To login use:

- user: `user@autodisc.com`
- password: `autodisc`

You should see the server list in the panel on the left. To see the data of a
table, unfold the `schema/Tables` property of the server. Then, right click on
the desired table and click "See data".

#### MongoExpress

MongoExpress is also a lightweight DB GUI but this time made for MongoDB.
Access the app at [http://localhost:8081](http://localhost:8081).
