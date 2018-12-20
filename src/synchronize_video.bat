REM Start times for cameras 1, 2, 3, 4, 6, 7, 8
REM [17.658, 10.274,  0.   , 14.112, 16.152,  1.104,  9.298]
REM round these off to nearest multiple of 1 / 30 to match frames @ 30 FPS

D:
REM this was the simple idea but it didn'e replicate the exact number of frames
REM cd D:\Dropbox\Harvard-AM-205-Basketball\Video\04_Trimmed_MPG
REM ffmpeg -i Camera1.mpg -ss 00:17.666 -t 02:26.333 -c copy Camera1_sync.mpg
REM ffmpeg -i Camera2.mpg -ss 00:10.266 -t 02:26.333 -c copy Camera2_sync.mpg
REM ffmpeg -i Camera3.mpg -ss 00:00.000 -t 02:26.333 -c copy Camera3_sync.mpg
REM ffmpeg -i Camera4.mpg -ss 00:14.091 -t 02:26.333 -c copy Camera4_sync.mpg
REM ffmpeg -i Camera6.mpg -ss 00:16.166 -t 02:26.333 -c copy Camera6_sync.mpg
REM ffmpeg -i Camera7.mpg -ss 00:01.100 -t 02:26.333 -c copy Camera7_sync.mpg
REM ffmpeg -i Camera8.mpg -ss 00:09.666 -t 02:26.333 -c copy Camera8_sync.mpg

REM Generate synchronized audio using the offsets
cd D:\Dropbox\Harvard-AM-205-Basketball\Video\05_Audio_WAV
ffmpeg -i Camera1.wav -ss 00:17.658 -t 02:26.333 -c copy Camera1_sync.wav
ffmpeg -i Camera2.wav -ss 00:10.274 -t 02:26.333 -c copy Camera2_sync.wav
ffmpeg -i Camera3.wav -ss 00:00.000 -t 02:26.333 -c copy Camera3_sync.wav
ffmpeg -i Camera4.wav -ss 00:14.112 -t 02:26.333 -c copy Camera4_sync.wav
ffmpeg -i Camera6.wav -ss 00:16.152 -t 02:26.333 -c copy Camera6_sync.wav
ffmpeg -i Camera7.wav -ss 00:01.104 -t 02:26.333 -c copy Camera7_sync.wav
ffmpeg -i Camera8.wav -ss 00:09.298 -t 02:26.333 -c copy Camera8_sync.wav

REM Assemble the synchronized frames into new mp4 streams with video only
cd D:\Dropbox\Harvard-AM-205-Basketball\Video\07_SyncFrames
ffmpeg -r 30 -f image2 -s 1920x1080 -i Camera1/Camera1_SyncFrame%05d.png -vcodec libx264 -crf 20 -pix_fmt yuv420p Camera1_sync_video.mp4
ffmpeg -r 30 -f image2 -s 1920x1080 -i Camera2/Camera2_SyncFrame%05d.png -vcodec libx264 -crf 20 -pix_fmt yuv420p Camera2_sync_video.mp4
ffmpeg -r 30 -f image2 -s 1920x1080 -i Camera3/Camera3_SyncFrame%05d.png -vcodec libx264 -crf 20 -pix_fmt yuv420p Camera3_sync_video.mp4
ffmpeg -r 30 -f image2 -s 1920x1080 -i Camera4/Camera4_SyncFrame%05d.png -vcodec libx264 -crf 20 -pix_fmt yuv420p Camera4_sync_video.mp4
ffmpeg -r 30 -f image2 -s 1920x1080 -i Camera6/Camera6_SyncFrame%05d.png -vcodec libx264 -crf 20 -pix_fmt yuv420p Camera6_sync_video.mp4
ffmpeg -r 30 -f image2 -s 1920x1080 -i Camera7/Camera7_SyncFrame%05d.png -vcodec libx264 -crf 20 -pix_fmt yuv420p Camera7_sync_video.mp4
ffmpeg -r 30 -f image2 -s 1920x1080 -i Camera8/Camera8_SyncFrame%05d.png -vcodec libx264 -crf 20 -pix_fmt yuv420p Camera8_sync_video.mp4

REM add audio to these mp4 streams
ffmpeg -i Camera1_sync_video.mp4 -i Camera1_sync.wav -c:v copy Camera1_sync.mp4
ffmpeg -i Camera2_sync_video.mp4 -i Camera2_sync.wav -c:v copy Camera2_sync.mp4
ffmpeg -i Camera3_sync_video.mp4 -i Camera3_sync.wav -c:v copy Camera3_sync.mp4
ffmpeg -i Camera4_sync_video.mp4 -i Camera4_sync.wav -c:v copy Camera4_sync.mp4
ffmpeg -i Camera6_sync_video.mp4 -i Camera6_sync.wav -c:v copy Camera6_sync.mp4
ffmpeg -i Camera7_sync_video.mp4 -i Camera7_sync.wav -c:v copy Camera7_sync.mp4
ffmpeg -i Camera8_sync_video.mp4 -i Camera8_sync.wav -c:v copy Camera8_sync.mp4

REM Assemble the tableau frames into new mp4 stream with video only
cd D:\Dropbox\Harvard-AM-205-Basketball\Video\12_Tableau
rem ffmpeg -r 30 -f image2 -s 3840x3240 -i Tableau%05d.png -vcodec libx264 -crf 20 -pix_fmt yuv420p Tableau_video.mp4
rem ffmpeg -r 30 -f image2 -s 3840x3240 -i Tableau%05d.png -vcodec libx264 -b:v 64k -crf 20 -pix_fmt yuv420p Tableau_video.mp4
ffmpeg -r 30 -f image2 -s 3840x3240 -i Tableau%05d.png -vcodec libx264 -b:v 64k -maxrate 128K -bufsize 128M -crf 20 -pix_fmt yuv420p Tableau_video.mp4
ffmpeg -i Tableau_video.mp4 -i Camera2_sync.wav -i Camera3_sync.wav -i Camera4_sync.wav -i Camera6_sync.wav -i Camera7_sync.wav -c:v copy Tableau.mp4