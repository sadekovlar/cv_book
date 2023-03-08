.PHONY : download install fullinstall

download:
	wget "https://cloclo.datacloudmail.ru/zip64/0XTDKqxs2k0wGrsITpC0yEgcMCqRb41z1p8caVEEoSRryAN7Ed1cGfp1Vs/data.zip"
	unzip data.zip
	rm -rf data.zip

install:
	pip install .

fullinstall: download install
