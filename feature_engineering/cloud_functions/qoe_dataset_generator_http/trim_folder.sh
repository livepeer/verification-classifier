for i in HD/*.mp4; do ffmpeg -ss 00:00:03 -i "$i" -t 00:00:10 -async 1 -c copy "${i%.*}-10s.mp4"; done

