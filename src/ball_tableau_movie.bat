D:
cd D:\Dropbox\Harvard-AM-205-Basketball\Video\13_BallTableau

ffmpeg -r 30 -f image2 -s 3840x3240 -i BallTableau%05d.png -vcodec libx264 -b:v 64k -maxrate 128K -bufsize 128M -crf 20 -pix_fmt yuv420p BallTableau_video.mp4
ffmpeg -i BallTableau_video.mp4 -i Camera2_sync.wav -i Camera3_sync.wav -i Camera4_sync.wav -i Camera6_sync.wav -i Camera7_sync.wav -c:v copy Tableau.mp4