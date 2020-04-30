for filename in $(gsutil ls gs://livepeer-qoe-sources/vimeo)
do
    # Define a timestamp function
    filename="${filename##*/}"
    echo "$(date)" "$filename";
    curl localhost:5000/verify -d '{"source":"https://storage.googleapis.com/livepeer-qoe-sources/vimeo/'$filename'","renditions":[{"uri":"https://storage.googleapis.com/livepeer-qoe-renditions/720_14/vimeo/'$filename'"}],"orchestratorID": "foo","model": "https://storage.googleapis.com/verification-models/verification-metamodel.tar.xz"}' -H 'Content-Type: application/json' &
    sleep 1
done;