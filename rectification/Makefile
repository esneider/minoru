.PHONY: all cameras clean

all:
	g++ -o calibrate calibrate.cpp `pkg-config --libs --cflags opencv` -std=c++11
	chmod a+x calibrate
quirks:
	sudo rmmod uvcvideo
	sudo modprobe uvcvideo quirks=128
clean:
	rm -f calibrate *.yml
