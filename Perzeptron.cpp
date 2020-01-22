#include "Perzeptron.h"
#include <iostream>
#include <ctime>


float scalar_prod(float* a, float* b, int l) {
	float result = 0;
	for (int i = 0; i < l; i++) {
		result += a[i] * b[i];
	}
	return result;
}

float activate_function(float a) {
	return 1 / (1 + exp(-a));
}

float learning_curve(float a) {
	return activate_function(a) * (1 - activate_function(a));
}

Perzeptron::Perzeptron(int n_in) {
	n = n_in+1;
	w = new float[n];
	w[0] = 1;
	for (int i = 1; i < n; i++) w[i] = ((float)rand())/RAND_MAX;
}

// backpropagation
void Perzeptron::train(float *zf, float dv, float eta_next) {
	float a, adjustment;
	a = w[0] + scalar_prod(w + 1, zf, n - 1);
	last_eta = eta_next * learning_curve(a);
	adjustment = last_eta * dv;
	w[0] += adjustment;
	for (int i = 1; i < n; i++) {
		w[i] += adjustment * zf[i - 1];
	}
}

float Perzeptron::eval(float* input) {
	float a = w[0] + scalar_prod(w+1, input, n - 1);
	std::cout << "t" << a << std::endl;
	last_eval = activate_function(a);
	return last_eval;
}

float Perzeptron::get_last_eval() {
	return last_eval;
}

float Perzeptron::get_last_eval_index(int at_index) {
	return last_eta * w[at_index+1];
}

