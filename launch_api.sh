docker build -f Dockerfile-api -t verifier-api:v1 . && docker run --volume="$(pwd)/stream":/stream -p 5000:5000 verifier-api:v1