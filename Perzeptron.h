#pragma once
class Perzeptron
{
	public:
		Perzeptron(int n_in);
		void train(float *zf, float dv, float eta_next);
		float eval(float* input);
		float get_last_eval();
		float get_last_eval_index(int index);
	private:
		float last_eval = 0, last_eta = 0;
		float* w;
		int n = 5;
};

