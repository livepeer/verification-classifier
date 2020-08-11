#!/bin/bash
docker rm verifier-api
docker build -f Dockerfile -t livepeer/verifier:latest .
docker build -f Dockerfile-api -t livepeer/verifier-api:latest .
docker run --volume="$(pwd)/stream":/stream -p 5000:5000 livepeer/verifier-api:latest