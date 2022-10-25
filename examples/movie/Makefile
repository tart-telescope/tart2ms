data:
	bash download_tart_data.sh

ms:
	tart2ms --phase_center_policy rephase-obs-midpoint --hdf ~/Downloads/tart_obs/tart_obs/*.hdf --ms tart_obs_hdf.ms  --telescope_name "KAT-7"

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
	ffmpeg -framerate 3 -i 'obs_%d.jpeg' \
		-c:v libx264 -pix_fmt yuv420p wsclean_tart_obs.mp4
