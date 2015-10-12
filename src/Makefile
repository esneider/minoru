CPP_FLAGS=stereo.cpp -I. -g -std=c++11 elas/*.cpp -Ielas/ -msse3 `pkg-config --libs --cflags opencv`

.PHONY: all quirks clean

all:
	# g++ -o calibrate calibrate.cpp ${CPP_FLAGS}
	g++ -o calibrate_from_file calibrate_from_file.cpp ${CPP_FLAGS}
	g++ -o rectify rectify.cpp ${CPP_FLAGS}
	chmod a+x calibrate_from_file rectify

quirks:
	sudo rmmod uvcvideo
	sudo modprobe uvcvideo quirks=128

clean:
	rm -f calibrate calibrate_from_file rectify *.yml
