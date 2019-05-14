cp ../../scripts/asset_processor/video_asset_processor.py imports/
cp ../../scripts/asset_processor/video_metrics.py imports/
gcloud beta functions deploy measure_asset_http --runtime python37 --trigger-http --memory 2048 --timeout 240