#include "Layer.h"
#include <iostream>


void Layer::init(int inputs, int outputs) {
	c_in = inputs;
	c_out = outputs;
	neurons = new Perzeptron*[c_out];
	for (int i = 0; i < c_out; i++) {
		neurons[i] = new Perzeptron(c_in);
	}
	weights = new float[inputs];
}

float* Layer::train(float* zf, float dv, float* prev_weights) {
	for (int i = 0; i < c_out; i++) {
		neurons[i]->train(zf, dv, prev_weights[i]);
	}
	calc_weights();
	return weights;
}

float* Layer::eval(float* input) {
	float* res = new float(c_out);
	for (int i = 0; i < c_out; i++) {
		res[i]=neurons[i]->eval(input);
	}
	return res;
}

float* Layer::get_last_eval() {
	float* res = new float(c_out);
	for (int i = 0; i < c_out; i++) {
		res[i] = neurons[i]->get_last_eval();
	}
	return res;
}

void Layer::calc_weights() {
	for (int i = 0; i < c_in; i++) {
		weights[i] = 0;
		for (int j = 0; j < c_out; j++) {
			weights[i] += neurons[j]->get_last_eval_index(i);
		}
		weights[i] /= c_out;
	}
}