TARGET=signal
DIR=flyby_${TARGET}
MS=${DIR}.ms

IMAGE=image
all:
# 	make flyby TARGET=${TARGET}
# 	make single_ms TARGET=${TARGET}
	make h5 -j `nproc` TARGET=${TARGET}
	make svg -j `nproc` TARGET=${TARGET}
	make png -j `nproc` TARGET=${TARGET}
	make webm -j `nproc` TARGET=${TARGET}

.PRECIOUS: $(h5files)
.PRECIOUS: $(svgfiles)
.PRECIOUS: $(pngfiles)

TART_OBS=${DIR}

# Get the last hours worth of data from the s3 bucket
flyby:
	rm -rf ${DIR}
	tart_get_archive_data --dir ${DIR} --target ${TARGET} --start 2023-12-01T00:25:00 --duration 5

single_ms:
	tart2ms  --hdf ${obsfiles} --clobber --ms ${MS}  --telescope_name "TART - signal"

NUMBER_OF_FIELDS=239

# Create a list of imaginary targets for each of the frames in the measurement set.
baselist := $(shell for i in `seq 0 ${NUMBER_OF_FIELDS}`; do echo "${DIR}/${IMAGE}_"; done)
fileroots := $(join $(baselist), $(shell seq -w 0 ${NUMBER_OF_FIELDS}))

h5files  := $(addsuffix .h5, $(fileroots))


SPHERE=--healpix --fov 170deg --res 1deg

#%.h5: ${MS}
#	$(eval FIELD := $(lastword $(subst _, ,$(basename $(notdir $@)))))
#	disko --ms ${MS} ${SPHERE} --field $(FIELD) --lasso --alpha 0.0015 --l1-ratio 0.02 --HDF $@

%.h5: ${MS}
	$(eval FIELD := $(lastword $(subst _, ,$(basename $(notdir $@)))))
	spotless --ms $< ${SPHERE} --field $(FIELD) --multimodel --HDF $@


# 2023-11-30T23:30:00


# Get the last hours worth of data from the s3 bucket
hour:
	rm -rf ${DIR}
	tart_get_archive_data --dir ${DIR} --start -60 --duration 60  --n 60


# --rephase "J153145-454621"
%.ms: %.hdf
	tart2ms  --hdf $< --clobber --ms $@  --telescope_name "KAT-7"


%.svg: %.h5
	disko_draw --show-sources --SVG $@  $<

 
%.png: %.svg
#	convert $< $@
	inkscape --export-width=1000 --export-area-drawing --export-filename=$@ --export-overwrite --export-type=png $<

obsfiles := $(wildcard ${TART_OBS}/obs_?????.hdf)

svgfiles := $(patsubst %.h5,%.svg,$(h5files))
pngfiles := $(patsubst %.svg,%.png,$(svgfiles))



h5:	$(h5files)
	echo "Done"

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

FFMPEG_OPT=-framerate 25 -pattern_type glob -i '${TART_OBS}/image_*.png' -vf scale=720:-1 -c:v libvpx-vp9 -b:v 1M -pix_fmt yuv420p

webm: $(pngfiles)
	ffmpeg ${FFMPEG_OPT} -pass 1 -an -f null /dev/null
	ffmpeg ${FFMPEG_OPT} -pass 2 -c:a libopus output_${TARGET}.webm
