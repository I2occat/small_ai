
#include "Perzeptron.h"
#pragma once
class Layer
{
	public:
		void init(int inputs, int outputs);
		float* train(float* zf, float dv, float* prev_weights);
		float* eval(float* input);
		float* get_last_eval();
	private:
		void calc_weights();
		int c_in;
		int c_out;
		Perzeptron** neurons;
		float* weights;
};

