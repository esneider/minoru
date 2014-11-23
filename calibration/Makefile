.PHONY: all run clean rectify

all:
	g++ -o calibrate calibrate.cpp `pkg-config --libs --cflags opencv`
	g++ -o rectify rectify.cpp `pkg-config --libs --cflags opencv`
	chmod a+x calibrate
run:
	./calibrate in_right.xml
	./calibrate in_left.xml
clean:
	rm -f calibrate out_right.xml out_left.xml
rectify:
	./rectify out_right.xml out_left.xml
