FROM --platform=linux/amd64 python:3.7.14-slim

RUN apt-get update -y
RUN apt-get install -y gcc

# AutoDiscServer dependency resolution
# NOTE: can only run in the build context provided in docker-compose.yml
COPY services/AutoDiscServer/flask_app/app.py /usr/src/services/flask_app/app.py
COPY services/AutoDiscServer/flask_app/requirements.txt /usr/src/services/flask_app/requirements.txt
COPY libs/auto_disc /usr/src/libs/auto_disc
WORKDIR /usr/src/libs/auto_disc
RUN pip install -e . 
RUN pip install -r /usr/src/services/flask_app/requirements.txt

COPY libs/utils /usr/src/libs/utils
WORKDIR /usr/src/libs/utils
RUN pip install -e . 

# nginx setup
RUN apt-get install -y nginx 
RUN rm /etc/nginx/sites-enabled/default
COPY services/AutoDiscServer/nginx/nginx.conf /etc/nginx/sites-available/flask
RUN ln -s /etc/nginx/sites-available/flask /etc/nginx/sites-enabled/flask

# dev tools
RUN apt-get install -y curl procps vim

# placed here to allowed efficient caching of src changes
COPY services/AutoDiscServer/flask_app/ /usr/src/services/flask_app/
# start daemons
VOLUME ["/usr/src/services", "/usr/src/libs"]
WORKDIR /usr/src/services/flask_app
#CMD gunicorn -w 1 -b 0.0.0.0:8000 app:app --daemon ; nginx -g "daemon off;"
CMD gunicorn -w 4 -b 0.0.0.0:8000 app:app --access-logfile '-' --log-level 'debug' & nginx -g "daemon off;"
