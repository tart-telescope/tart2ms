data:
	TARGET=rhodes bash download_tart_data.sh

TART_OBS=.
%.ms: %.hdf
	tart2ms --rephase obs-midpoint --single-field --hdf $< --clobber --ms $@  --telescope_name "KAT-7"

disko:
	disko --file obs_1.json --healpix --fov 180deg --res 1deg --lasso --alpha 0.0025 --l1-ratio 0.02 --show-sources --SVG


%.h5: %.ms
	disko --ms $< --healpix --fov 180deg --res 1deg --lasso --alpha 0.0025 --l1-ratio 0.02 --HDF $@ 

%.svg: %.h5
	disko_draw --SVG $@  $<

# %.svg: %.ms
# 	spotless --ms $< --healpix --fov 180deg --multimodel --res 1deg --SVG
# 	mv spotless_2022*.svg $@

%.png: %.svg
	convert $< $@

msfiles  := $(patsubst %.hdf,%.ms,$(wildcard ${TART_OBS}/obs_*.hdf))
svgfiles := $(patsubst %.hdf,%.svg,$(wildcard ${TART_OBS}/obs_*.hdf))
pngfiles := $(patsubst %.hdf,%.png,$(wildcard ${TART_OBS}/obs_*.hdf))


ms:	$(msfiles)
	echo "Dne"
svg:	$(svgfiles)
	echo "Done"
png:	$(pngfiles)
	echo "Done"

clean:
	rm -f ${svgfiles}
	rm -f ${pngfiles}
	
mov: $(pngfiles)
	ffmpeg -framerate 3  -i 'obs_%d.png' \
		-c:v libx264 -pix_fmt yuv420p tim_tart_obs.mp4

wsmov: $(pngfiles)
	ffmpeg -framerate 3 -i 'obs_%d.png' \
		-c:v libvpx-vp9 -b:v 800k -pix_fmt yuv420p disko_scp_tart_obs.webm
