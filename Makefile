.PHONY : download install fullinstall

download:
	wget "https://cloclo.datacloudmail.ru/zip64/V7eVPRE2ArgWYvm1S3EPI4ckr8WGSwFKbAW0u9Nk2Mb5YRcNtM79HweQNY/data.zip"
	unzip data.zip
	rm -rf data.zip

install:
	pip install .

fullinstall: download install
