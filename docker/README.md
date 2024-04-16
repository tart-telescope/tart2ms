## Docker container for testing tart2ms

This creates a docker container that makes life easy for testing. Just type

    make
    
in this directory, and through the magic of docker you'll be able to test using CASA. The local directory is available as /remote in the docker container.

Some commands to use are:

=== casabrowser ===

    cd remote
    casabrowser

This lets you look at the contents of a measurement set.

=== casaviewer ===

    cd remote
    casaviewer

Lets you view a fits file.
