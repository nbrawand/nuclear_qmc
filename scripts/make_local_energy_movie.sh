ffmpeg -y -framerate 1 -i local_energy_%05d.png output.mp4
#ffmpeg -framerate 1/5 -i *%03d.png -c:v libx264 -r 30 -pix_fmt yuv420p out.mp4
