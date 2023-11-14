TARGET=rhodes
DIR=data_${TARGET}


data:
	TARGET=rhodes bash download_tart_data.sh

SPHERE=--healpix --fov 180deg --res 1deg
VENV=~/.tartvenv/bin

TART_OBS=${DIR}
%.ms: %.hdf
	${VENV}/tart2ms --rephase obs-midpoint --single-field --hdf $< --clobber --ms $@  --telescope_name "KAT-7"



%.svg: %.h5
	${VENV}/disko_draw --SVG $@  $<

%.h5: %.ms
	${VENV}/disko --ms $< ${SPHERE} --lasso --alpha 0.0025 --l1-ratio 0.02 --HDF $@ 


# %.h5: %.ms
# 	${VENV}/spotless --ms $< ${SPHERE} --multimodel --HDF $@
# 
%.png: %.svg
	convert $< $@

msfiles  := $(patsubst %.hdf,%.ms,$(wildcard ${TART_OBS}/obs_*.hdf))
svgfiles := $(patsubst %.hdf,%.svg,$(wildcard ${TART_OBS}/obs_*.hdf))
pngfiles := $(patsubst %.hdf,%.png,$(wildcard ${TART_OBS}/obs_*.hdf))
h5files := $(patsubst %.hdf,%.h5,$(wildcard ${TART_OBS}/obs_*.hdf))


ms:	$(msfiles)
	echo "Dne"
svg:	$(svgfiles)
	echo "Done"
png:	$(pngfiles)
	echo "Done"

clean:
	rm -f ${svgfiles}
	rm -f ${pngfiles}
	rm -f ${h5files}
	
mov: $(pngfiles)
	cd ${TART_OBS} && ffmpeg -framerate 3  -i 'obs_%d.png' \
		-c:v libx264 -pix_fmt yuv420p tim_tart_obs.mp4

FFMPEG_OPT=-framerate 3 -i '${TART_OBS}/obs_%d.png' -vf scale=420:-1 -c:v libvpx-vp9 -b:v 1M -pix_fmt yuv420p

webm: $(pngfiles)
	ffmpeg ${FFMPEG_OPT} -pass 1 -an -f null /dev/null
	ffmpeg ${FFMPEG_OPT} -pass 2 -c:a libopus output.webm
