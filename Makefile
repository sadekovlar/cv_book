.PHONY : download install fullinstall test

download:
	wget "https://cloclo.datacloudmail.ru/zip64/V7eVPRE2ArgWYvm1S3EPI4ckr8WGSwFKbAW0u9Nk2Mb5YRcNtM79HweQNY/data.zip"
	unzip data.zip
	rm -rf data.zip

install:
	pip install .

fullinstall: download install

test:
	pip install ./srccam
	pip install pytest "opencv-python-headless>=3.4"
	python -m pytest tests/test_srccam_smoke.py -q
