.PHONY: all run clean

all:
	g++ -o calibrate calibrate.cpp `pkg-config --libs --cflags opencv`
	chmod a+x calibrate
run:
	./calibrate in_right.xml
	./calibrate in_left.xml
clean:
	rm -f calibrate out_right.xml out_left.xml
