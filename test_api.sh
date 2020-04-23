for filename in $(ls stream/sources)
do
    # Define a timestamp function
    filename="${filename##*/}"
    curl localhost:5000/verify -d '{"source":"stream/sources/'$filename'","renditions":[{"uri":"stream/720p_black_and_white/_AN9U4D1Cww.mp4"},{"uri":"stream/240p_flip_horizontal/'$filename'"}],"orchestratorID": "foo","model": "https://storage.googleapis.com/verification-models/verification.tar.xz"}' -H 'Content-Type: application/json' &
    echo "$(date)" "$filename";
    sleep 1
done;