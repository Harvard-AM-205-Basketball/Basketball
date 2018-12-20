D:
cd D:\Dropbox\Harvard-AM-205-Basketball\Video\13_BallTableau

REM Make a movie of the tableau
ffmpeg -r 30 -f image2 -s 3840x3240 -i BallTableau%05d.png -vcodec libx264 -b:v 128k -crf 20 -pix_fmt yuv420p BallTableau_video.mp4
ffmpeg -i BallTableau_video.mp4 -i Camera2_sync.wav -i Camera3_sync.wav -i Camera4_sync.wav -i Camera6_sync.wav -i Camera7_sync.wav -c:v copy BallTableau.mp4

cd D:\Dropbox\Harvard-AM-205-Basketball\Video\11_BallFrames
ffmpeg -r 30 -f image2 -s 1920x1080 -i Camera1/Camera1_BallFrame%05d.png -vcodec libx264 -crf 20 -pix_fmt yuv420p Camera1_ball_video.mp4
ffmpeg -r 30 -f image2 -s 1920x1080 -i Camera2/Camera2_BallFrame%05d.png -vcodec libx264 -crf 20 -pix_fmt yuv420p Camera2_ball_video.mp4
ffmpeg -r 30 -f image2 -s 1920x1080 -i Camera3/Camera3_BallFrame%05d.png -vcodec libx264 -crf 20 -pix_fmt yuv420p Camera3_ball_video.mp4
ffmpeg -r 30 -f image2 -s 1920x1080 -i Camera4/Camera4_BallFrame%05d.png -vcodec libx264 -crf 20 -pix_fmt yuv420p Camera4_ball_video.mp4
ffmpeg -r 30 -f image2 -s 1920x1080 -i Camera6/Camera6_BallFrame%05d.png -vcodec libx264 -crf 20 -pix_fmt yuv420p Camera6_ball_video.mp4
ffmpeg -r 30 -f image2 -s 1920x1080 -i Camera7/Camera8_BallFrame%05d.png -vcodec libx264 -crf 20 -pix_fmt yuv420p Camera7_ball_video.mp4
ffmpeg -r 30 -f image2 -s 1920x1080 -i Camera8/Camera9_BallFrame%05d.png -vcodec libx264 -crf 20 -pix_fmt yuv420p Camera8_ball_video.mp4

REM add audio to these mp4 streams
ffmpeg -i Camera1_ball_video.mp4 -i Camera1_sync.wav -c:v copy Camera1_ball.mp4
ffmpeg -i Camera2_ball_video.mp4 -i Camera1_sync.wav -c:v copy Camera2_ball.mp4
ffmpeg -i Camera3_ball_video.mp4 -i Camera1_sync.wav -c:v copy Camera3_ball.mp4
ffmpeg -i Camera4_ball_video.mp4 -i Camera1_sync.wav -c:v copy Camera4_ball.mp4
ffmpeg -i Camera6_ball_video.mp4 -i Camera1_sync.wav -c:v copy Camera6_ball.mp4
ffmpeg -i Camera7_ball_video.mp4 -i Camera1_sync.wav -c:v copy Camera7_ball.mp4
ffmpeg -i Camera8_ball_video.mp4 -i Camera1_sync.wav -c:v copy Camera8_ball.mp4
