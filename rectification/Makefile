.PHONY: all cameras clean

all:
	g++ -msse3 -o cameras camerassift.cpp _libelas/*.cpp -I_libelas `pkg-config --libs --cflags opencv`
	g++ -o calibrate calibrate.cpp `pkg-config --libs --cflags opencv`
	chmod a+x cameras calibrate
cameras:
	./cameras
calibrate:
	./calibrate 1 out1.yml
	./calibrate 2 out2.yml
quirks:
	sudo rmmod uvcvideo
	sudo modprobe uvcvideo quirks=128
clean:
	rm -f cameras calibrate *.yml
