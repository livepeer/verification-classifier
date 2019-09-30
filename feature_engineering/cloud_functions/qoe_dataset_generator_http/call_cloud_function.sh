#!/bin/bash

for filename in $(gsutil ls gs://livepeer-qoe-sources/HD)
do
    # Define a timestamp function
 
    filename="${filename##*/}"
    resolution_array=( 1080 720 480 384 288 144 )
    for resolution in "${resolution_array[@]}"
    do
        gcloud functions call qoe_dataset_generator_http --data '{"name":"'"$filename"'" , "resolution":"'"$resolution"'"}' &

        echo "$(date)" "$filename" "$resolution";
        sleep 16
    done
done;