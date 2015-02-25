.PHONY: all quirks clean

all:
	g++ -o calibrate calibrate.cpp -I. `pkg-config --libs --cflags opencv` -std=c++11
	g++ -o rectify rectify.cpp elas/*.cpp -I. -Ielas/ -msse3 `pkg-config --libs --cflags opencv` -std=c++11
	chmod a+x calibrate rectify
quirks:
	sudo rmmod uvcvideo
	sudo modprobe uvcvideo quirks=128
clean:
	rm -f calibrate rectify *.yml
