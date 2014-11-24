.PHONY: all cameras clean

all:
	g++ -o cameras cameras.cpp `pkg-config --libs --cflags opencv`
	chmod a+x cameras
cameras:
	./cameras
quirks:
	sudo rmmod uvcvideo
	sudo modprobe uvcvideo quirks=128
clean:
	rm -f cameras
