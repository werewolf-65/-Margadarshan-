ffmpeg -start_number 0 -framerate 15 -i ./out/processed_%d.png -s:v 1280:720 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ./out_cap.mp4
