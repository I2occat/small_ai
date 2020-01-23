#pragma once
class Perzeptron
{
	public:
		Perzeptron(int n_in);
		void train(float *zf, float dv, float eta_next);
		float eval(float* input);
		float get_last_eval();
		float get_last_eval_index(int index);
		void set_weights(float* weights);
	private:
		float last_eval = 0;
		float* w;
		float* last_eta;
		int n = 5;
};

