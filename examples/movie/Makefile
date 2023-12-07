TARGET=signal
DIR=flyby_${TARGET}

all:
	make ms TARGET=${TARGET}
	make h5 TARGET=${TARGET}
	make svg TARGET=${TARGET}
	make png TARGET=${TARGET}
	make webm TARGET=${TARGET}

# Get the last hours worth of data from the s3 bucket
flyby:
	rm -rf ${DIR}
	tart_get_archive_data --dir ${DIR} --target ${TARGET} --start 2023-12-01T00:25:00 --duration 10

single_ms:
	tart2ms  --hdf ${obsfiles} --clobber --ms flyby.ms  --telescope_name "TART - signal"

images:
	for number in `seq -w 0 540`; do \
		disko --ms flyby.ms ${SPHERE} --field $$number --lasso --alpha 0.0025 --l1-ratio 0.02 --HDF ${DIR}/image_$$number.h5 ; \
	done

# 2023-11-30T23:30:00


# Get the last hours worth of data from the s3 bucket
hour:
	rm -rf ${DIR}
	tart_get_archive_data --dir ${DIR} --start -60 --duration 60  --n 60

data:
	TARGET=${TARGET} bash download_tart_data.sh

SPHERE=--healpix --fov 170deg --res 1deg


TART_OBS=${DIR}

# --rephase "J153145-454621"
%.ms: %.hdf
	tart2ms  --hdf $< --clobber --ms $@  --telescope_name "KAT-7"


%.svg: %.h5
	disko_draw --max 16.0 --show-sources --SVG $@  $<

%.h5: %.ms
	for number in `seq -w 0 59`; do \
		disko --ms $< ${SPHERE} --field $$number --lasso --alpha 0.0025 --l1-ratio 0.02 --HDF $*$$number.h5 ; \
	done
	touch $@


#%.h5: %.ms
#	spotless --ms $< ${SPHERE} --multimodel --HDF $@
 
%.png: %.svg
#	convert $< $@
	inkscape --export-width=1000 --export-area-drawing --export-filename=$@ --export-overwrite --export-type=png $<

obsfiles := $(wildcard ${TART_OBS}/obs_?????.hdf)
msfiles  := $(patsubst %.hdf,%.ms,$(wildcard ${TART_OBS}/obs_?????.hdf))
h5files  := $(patsubst %.ms,%.h5,$(wildcard ${TART_OBS}/obs_?????.ms))
svgfiles := $(patsubst %.h5,%.svg,$(wildcard ${TART_OBS}/image_*.h5))
pngfiles := $(patsubst %.svg,%.png,$(wildcard ${TART_OBS}/obs_*.svg))


ms:	$(msfiles)
	echo "Done"
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

FFMPEG_OPT=-framerate 25 -pattern_type glob -i '${TART_OBS}/obs_*.png' -vf scale=720:-1 -c:v libvpx-vp9 -b:v 1M -pix_fmt yuv420p

webm: 
	ffmpeg ${FFMPEG_OPT} -pass 1 -an -f null /dev/null
	ffmpeg ${FFMPEG_OPT} -pass 2 -c:a libopus output_${TARGET}.webm
