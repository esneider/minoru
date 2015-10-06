CPP_FLAGS=stereo.cpp -I. `pkg-config --libs --cflags opencv` -std=c++11

.PHONY: all quirks clean

all:
	g++ -o calibrate calibrate.cpp ${CPP_FLAGS}
	g++ -o rectify rectify.cpp ${CPP_FLAGS}
	g++ -o rectify_from_file rectify_from_file.cpp ${CPP_FLAGS}
	chmod a+x calibrate rectify
quirks:
	sudo rmmod uvcvideo
	sudo modprobe uvcvideo quirks=128
clean:
	rm -f calibrate rectify *.yml
