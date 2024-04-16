# Build a movie of TART observations

There is a script ```download_tart_data.sh``` that will download 30 observations 2 minutes apart. The observations will be downloaded as both JSON files and HDF visibility files.


```
    bash download_tart_data.sh
```

The observation JSON files are called obs_n.json. The visibility files are timestamped.


## The JSON way.

### Creating Images 

This is done using disko which creates an SVG all-sky image, using a makefile rewrite rule.

```
    disko --file $< --healpix --fov 180deg --res 1deg --lasso --alpha 0.0025 --l1-ratio 0.02 --show-sources --SVG
```

This is converted to .png using another rewrite rule.

### Making the movie.

This is made using ffmpeg. See the makefile for details.

## The measurement set way

This will create a measurement set, rephased to the zenith.
