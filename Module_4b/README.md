# SuperPoint
https://github.com/sadekovlar/super_point_relative_navi
# некоторые команды для вызова

python demo_superpoint.py ./video/get.130.151.left.avi
python matching.py ./video/get.130.151.left.avi
python relative_pose.py

# SuperGlue
https://github.com/magicleap/SuperGluePretrainedNetwork
# некоторые команды для вызова

python demo_superglue.py
python demo_superglue.py --input assets/phototourism_sample_images/ --output_dir dump_demo_sequence --resize 320 240
python demo_superglue.py --input assets/freiburg_sequence/ --output_dir dump_demo_sequence --resize 320 240
python matching.py
# Сравнение SuperPoint и SIFT на двух видео из data/SIFT
# (один раз: cd Module_4b && git clone https://github.com/magicleap/SuperPointPretrainedNetwork.git; pip install torch)
python Module_4b/superpoint_video_match.py
python Module_4b/superpoint_video_match.py --offset 10 --show
# результат: data/SIFT/out/superpoint_vs_sift.mp4, superpoint_vs_sift.csv, superpoint_vs_sift_first.jpg
