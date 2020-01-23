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
	last_eta = new float[n_in];
	n = n_in+1;
	w = new float[n];
	w[0] = 1;
	for (int i = 1; i < n; i++) w[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

void Perzeptron::set_weights(float* weights) {
	for (int i = 0; i < n; i++) {
		w[i] = weights[i];
	}
}

// backpropagation
void Perzeptron::train(float *zf, float dv, float eta_next) {
	float a,eta, adjustment;
	a = w[0] + scalar_prod(w + 1, zf, n - 1);
	eta = eta_next * learning_curve(a);
	adjustment = eta * dv;
	w[0] += adjustment;
	for (int i = 1; i < n; i++) {
		last_eta[i - 1] = w[i] * eta;
		w[i] += adjustment * zf[i - 1];
	}
}

float Perzeptron::eval(float* input) {
	float a = w[0] + scalar_prod(w+1, input, n - 1);
	last_eval = activate_function(a);
	return last_eval;
}

float Perzeptron::get_last_eval() {
	return last_eval;
}

float Perzeptron::get_last_eval_index(int at_index) {
	return last_eta[at_index];
}

