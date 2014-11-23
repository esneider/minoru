.PHONY: all cameras text clean

all:
	g++ -o cameras cameras.cpp `pkg-config --libs --cflags opencv`
	g++ -o text text.cpp `pkg-config --libs --cflags opencv`
	chmod a+x cameras text
cameras:
	./cameras
text:
	./text
quirks:
	sudo rmmod uvcvideo
	sudo modprobe uvcvideo quirks=128
clean:
	rm -f cameras text
