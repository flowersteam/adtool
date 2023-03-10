#!/bin/bash

chmod 777 -R services/JupyterLab/Notebooks/
docker compose -f services/docker-compose.yml --profile prod up

# saves docker logs on the disk
source services/.env
mkdir -p ${LOG_FILE}
chmod 754 -R ${LOG_FILE}
currentTime=$(date '+%F_%H:%M:%S')

for value in app-db app-db-api expe-db expe-db-api autodisc-server jupyter app gateway; do
    mkdir -p ${LOG_FILE}/${value} && touch "${LOG_FILE}/${value}/${currentTime}.log"
    docker compose -f services/docker-compose.yml logs ${value} >"${LOG_FILE}/${value}/${currentTime}.log"
    echo "" >$(docker inspect --format='{{.LogPath}}' services-${value}-1)
done
