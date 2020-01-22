// FirstKi.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und endet die Ausführung des Programms.
//

#include <iostream>
#include "Layer.h"
#include "FirstKi.h"

void print_vector(float* v, int n) {
	std::cout << "|" << v[0];
	for (int j = 1; j < n; j++) {
		std::cout << "," << v[j];
	}
	std::cout << "|\r\n";
}

void print_matrix(float** a, int n, int m) {
	for (int i = 0; i < m; i++) {
		print_vector(a[i], n);
	}
}

int main()
{
	// init
	float** zf = new float* [4]; // Zielfunktion
	float* v = new float[4];		// Values
	for (int i = 0; i < 4; i++) {
		zf[i] = new float[2];
		zf[i][0] = (float)(i > 1 ? 1 : 0);
		zf[i][1] = (float)(i % 2 == 1 ? 1 : 0);
	}
	v[0] = 1;
	v[1] = 0;
	v[2] = 0;
	v[3] = 0;

	Layer* layer = new Layer();
	Layer* layer_out = new Layer();
	layer->init(2,2);
	layer_out->init(2,1);
	float* prev_weights = new float[3];
	for (int i = 0; i < 3; i++) {
		prev_weights[i] = 2;
	}
	for (int j = 0; j < 40; j++) {
		for (int i = 0; i < 4; i++) {
			// evaluate current state
			float result = layer_out->eval(layer->eval(zf[i]))[0];

			prev_weights = layer_out->train(layer->get_last_eval(), v[i] - result, prev_weights);
			layer->train(zf[i], v[i] - result, prev_weights);
		}
	}
	std::cout << "zf: \n";
	print_matrix(zf, 2, 4);
	std::cout << "\n";
	std::cout << "0: " << layer_out->eval(layer->eval(zf[0]))[0] << std::endl;
	std::cout << "1: " << layer_out->eval(layer->eval(zf[1]))[0] << std::endl;
	std::cout << "2: " << layer_out->eval(layer->eval(zf[2]))[0] << std::endl;
	std::cout << "3: " << layer_out->eval(layer->eval(zf[3]))[0] << std::endl;
}