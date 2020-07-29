# Verification API

The verifier API exposes video verification capabilities through HTTP REST endpoints. 

## API endpoints
### /verify

#### 
**Method:** `POST`  
**Content-type:** `application/json` or `multipart/form-data`  
**Description:**  
This function performs verification of renditions of source video. There are two use cases:  
1. video files are accessible to the server

    - content-type of the request should be application/json
2. video files are passed in request body (like a browser)

    - content-type of the request should be multipart/form-data
    - parameters should be passed as JSON in form's single 'json' field
    - files should be passed as multipart data, there should be correspondence between 'source', 'uri' fields in parameters and file names. Name fields of files should be unique.  
            
**Parameters:**  
```
{
  "orchestrator_id": "string, the ID of the orchestrator responsible of transcoding",
  "source": "string, an URI or name of the source video",
  "renditions": [
    {
      "uri": "string, a name or URI of rendition",
      "resolution": {
        "height": "integer, vertical dimension, in pixels",
        "width": "integer, horizontal dimension, in pixels"
      },
      "frame_rate": "float, expected FPS, with a special value of 0, which means 'same as source video'",
      "pixels": "integer, The number of expected total pixels (height x width x number of frames)"
    }
  ]
}
```
**Returns:**
```
{
  "orchestrator_id": "string, The ID of the orchestrator responsible of transcoding",
  "results": [
    {
      "audio_available": "boolean, whether file contained audio stream",
      "fps": "float, actual rendition framerate",
      "width": "integer, rendition frame width",
      "height": "integer, rendition frame height",
      "ocsvm_dist": "float, distance to separating plane of OCSVM classifier for this rendition, negative value represents an outlier (tampered rendition)",
      "tamper": "integer, 0 or 1, binary classification result, 1 if rendition predicted to be tampered, 0 otherwise",
      "tamper_sl": "integer, 0 or 1, binary classification result from supervised CatBoost model",
      "tamper_ul": "integer, 0 or 1, binary classification result from unsupervised OCSVM anomaly detector",
      "uri": "string, rendition URI",
      "video_available": "boolean, whether file contained a video stream"
    }
  ],
  "source": "string, source video URI"
}
``` 

## Running a docker image

To build the image and create a container, run the following bash script located in the root of the project:

```
./launch_api.sh
```

This will create and run a Docker image with API exposed on the port 5000.

## Examples

### Example (URI or shared volume path)

A sample call to the API is provided below:

*Request (remote assets)*

```

 curl localhost:5000/verify -d '{
                                "source": "https://storage.googleapis.com/livepeer-verifier-renditions/480p/-3MYFnEaYu4.mp4",
                                "renditions": [
                                                {
                                                    "uri": "https://storage.googleapis.com/livepeer-verifier-renditions/144p_black_and_white/-3MYFnEaYu4.mp4"
                                                },
                                                {
                                                    "uri": "https://storage.googleapis.com/livepeer-verifier-renditions/144p/-3MYFnEaYu4.mp4",
                                                    "resolution":{
                                                        "height":"144",
                                                        "width":"256"},
                                                    "frame_rate": "24",
                                                    "pixels":"1034500"
                                                }
                                            ],
                                "orchestratorID": "foo"
                                }' -H 'Content-Type: application/json'
```

*Response (remote assets)*

```

{"orchestrator_id":"foo",
"results":[
    {
            "video_available":true,
            "tamper":-1.195989,
            "uri":"https://storage.googleapis.com/livepeer-verifier-renditions/144p_black_and_white/-3MYFnEaYu4.mp4"
    },
    {
            "video_available":true,
            "frame_rate":false,
            "pixels":"1034500",
            "pixels_post_verification":0.09354202835648148,
            "pixels_pre_verification":127119360.0,
            "resolution":
            {
                "height":"144",
                "height_post_verification":1.0,
                "height_pre_verification":1.0,
                "width":"256",
                "width_post_verification":1.0,
                "width_pre_verification":1.0
            },
            "tamper":1.219913,
            "uri":"https://storage.googleapis.com/livepeer-verifier-renditions/144p/-3MYFnEaYu4.mp4"
    }],
    "source":"https://storage.googleapis.com/livepeer-verifier-renditions/480p/-3MYFnEaYu4.mp4"}

```

*Request (local assets)*

```

curl localhost:5000/verify -d '{
    "source": "stream/sources/1HWSFYQXa1Q.mp4",
    "renditions": [
        {
            "uri": "stream/144p_black_and_white/1HWSFYQXa1Q.mp4"
        },
        {
            "uri": "stream/144p/1HWSFYQXa1Q.mp4",
            "resolution":{
                "height":"144",
                "width":"256"
                },
            "frame_rate": "24",
            "pixels":"1034500"
        }
        ],
        "orchestratorID": "foo"
        }' -H 'Content-Type: application/json'

```

*Response (local assets)*
```
{
  "model": "https://storage.googleapis.com/verification-models/verification.tar.xz",
  "orchestrator_id": "foo",
  "results": [
    {
      "audio_available": false,
      "ocsvm_dist": -0.04083180936940067,
      "ssim_pred": 0.6080637397913853,
      "tamper_meta": -1,
      "tamper_sl": -1,
      "tamper_ul": -1,
      "uri": "stream/144p_black_and_white/1HWSFYQXa1Q.mp4",
      "video_available": true
    },
    {
      "audio_available": false,
      "frame_rate": false,
      "ocsvm_dist": 0.06808371913784983,
      "pixels": "1034500",
      "pixels_post_verification": 2.55114622790404,
      "pixels_pre_verification": 127119360,
      "resolution": {
        "height": "144",
        "height_post_verification": 1,
        "height_pre_verification": 1,
        "width": "256",
        "width_post_verification": 1,
        "width_pre_verification": 1
      },
      "ssim_pred": 0.6214110237850428,
      "tamper_meta": -1,
      "tamper_sl": -1,
      "tamper_ul": 1,
      "uri": "stream/144p/1HWSFYQXa1Q.mp4",
      "video_available": true
    }
  ],
  "source": "stream/sources/1HWSFYQXa1Q.mp4"
}

```

### Example (upload files in the query)
#### Request
Note: 
- filename parameters set explicitly to values used in URIs
- JSON parameters are passed in `json` form field
- file form fields have unique names (file1, file2) 
```
curl localhost:5000/verify -F 'file1=@../data/renditions/1080p/0fIdY5IAnhY_60.mp4;filename=1080_0fIdY5IAnhY_60.mp4' \
                           -F 'file2=@../data/renditions/720p/0fIdY5IAnhY_60.mp4;filename=720_0fIdY5IAnhY_60.mp4' \
                           -F 'json={
                                "source": "1080_0fIdY5IAnhY_60.mp4",
                                "renditions": [
                                                {
                                                    "uri": "720_0fIdY5IAnhY_60.mp4"
                                                }
                                            ],
                                "orchestratorID": "foo"
                                }'
```
#### Response
```
{
  "model": "http://storage.googleapis.com/verification-models/verification-metamodel-fps2.tar.xz",
  "orchestrator_id": "foo",
  "results": [
    {
      "audio_available": false,
      "fps": 60,
      "height": 720,
      "ocsvm_dist": 0.028662416537303254,
      "ssim_pred": 0.9728838728836663,
      "tamper": 0,
      "tamper_sl": 0,
      "tamper_ul": 1,
      "uri": "/tmp/d0424e5c79c9401d893d6f2b8e87dfc2/720_0fIdY5IAnhY_60.mp4",
      "video_available": true,
      "width": 1280
    }
  ],
  "source": "/tmp/d0424e5c79c9401d893d6f2b8e87dfc2/1080_0fIdY5IAnhY_60.mp4"
}
```