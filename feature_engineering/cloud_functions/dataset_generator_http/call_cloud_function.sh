#!/bin/bash

for filename in $(gsutil ls gs://livepeer-verifier-originals/)
do
    # Define a timestamp function
 
    filename="${filename##*/}"
    gcloud functions call dataset_generator_http --data '{"name":"'"$filename"'"}' &

    echo "$(date)" "$filename";
    sleep 7
done;