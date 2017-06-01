all:
	g++ -Wall -O3 -std=c++11 nb_classify.cpp -o nb_classify

clean:
	rm nb_classify
