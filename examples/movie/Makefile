data:
	bash download_tart_data.sh

TART_OBS=/home/tim/tart_obs
ms:
	tart2ms --rephase obs-midpoint --single-field --hdf ${TART_OBS}/*.hdf --clobber --ms tart_obs_hdf.ms  --telescope_name "KAT-7"

disko:
	disko --file obs_1.json --healpix --fov 180deg --res 1deg --lasso --alpha 0.0025 --l1-ratio 0.02 --show-sources --SVG


%.svg: %.json
	disko --file $< --healpix --fov 180deg --res 1deg --lasso --alpha 0.0025 --l1-ratio 0.02 --show-sources --SVG
	mv disko_*.svg $@

%.png: %.svg
	convert $< $@

pngfiles := $(patsubst %.json,%.png,$(wildcard obs_*.json))

mov: $(pngfiles)
	ffmpeg -framerate 3 -i 'obs_%d.png' \
		-c:v libx264 -pix_fmt yuv420p tim_tart_obs.mp4

wsmov: $(pngfiles)
	ffmpeg -framerate 3 -i 'obs_hdf_%d.jpeg' \
		-c:v libvpx-vp9 -b:v 800k -pix_fmt yuv420p disko_scp_tart_obs.webm
