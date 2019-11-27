cp ../../../scripts/asset_processor/video_asset_processor.py imports/
cp ../../../scripts/asset_processor/video_metrics.py imports/
gcloud functions deploy create_renditions_bucket_event --runtime python37 --trigger-resource livepeer-qoe-renditions-params --trigger-event google.storage.object.finalize --memory 2048 --verbosity debug --timeout 540
rm imports/video_asset_processor.py 
rm imports/video_metrics.py