.PHONY: all cameras clean

all:
	g++ -o calibrate calibrate.cpp `pkg-config --libs --cflags opencv`
	chmod a+x calibrate
quirks:
	sudo rmmod uvcvideo
	sudo modprobe uvcvideo quirks=128
clean:
	rm -f calibrate *.yml
