
#include "Perzeptron.h"
#pragma once
class Layer
{
	public:
		Layer(int inputs, int outputs);
		float* train(float* zf, float dv, float* prev_weights);
		float* eval(float* input);
		float* get_last_eval();
		void set_weights(float** weights);
	private:
		void calc_weights();
		int c_in;
		int c_out;
		Perzeptron** neurons;
		float* weights;
};

