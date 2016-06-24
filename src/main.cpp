/**
 * @file main.cpp
 * @brief simple mlp
 * @author k.ohta
 * @date 2016/06/23
 */

#include <memory>
#include <iostream>
#include <fstream>
#include <random>
#include <ctime>
#include <iomanip>
#include <algorithm>

class img_info;

void parse_idx(std::ifstream * const ifs_src, std::ifstream * const ifs_label, img_info * const obj_img_info);
template <typename T> void test(T ** const w_2, T ** const w_3, T * const u_2, T * const u_3, T * const d, T * const z, T * const y, T * const delta_2, T * const delta_3,
	img_info * test_img_info, std::ifstream * const ifs_src, std::ifstream * const ifs_label);
const int SRC_HEADER_OFFSET = 0x10;
const int LABEL_HEADER_OFFSET = 0x08;
const int IDX_OFFSET_NUMBER_OF_IMAGES = 0x04;
const int IDX_OFFSET_NUMBER_OF_ROWS = 0x08;
const int IDX_OFFSET_NUMBER_OF_COLUMNS = 0x0c;
const int MNIST_NUMBER_OF_OUTPUT_CLASS = 10;
const int HIDDEN_LAYER_UNIT = 100;
const int BIAS_SIZE = 1;
const double LEARNING_RATE_WEIGHT = 0.005;
const double LEARNING_RATE_BIAS = 0.01;

class img_info {
public :
	img_info() {};
	void set_param(int _n_img, int _width, int _height, int _n_output_class)
	{
		n_img = _n_img;
		n_output_class = _n_output_class;
		size = _width * _height;
	}
	int get_n_img() const { return n_img; }
	int get_size() const { return size; }
	int get_n_output_class() const { return n_output_class; }
private :
	int n_img;
	int size;
	int n_output_class;
};

template <class T> T ReLU(T val) {
	return val > static_cast<T>(0) ? val : static_cast<T>(0);
}

template <class T> void soft_max(T * const u, T * const y ,int n_output_class) {
	T sum = static_cast<T>(0);
	for (auto k = 0; k < n_output_class; k++)
		sum += std::exp(u[k + BIAS_SIZE]);
	for (auto k = 0; k < n_output_class; k++)
		y[k + BIAS_SIZE] = std::exp(u[k + BIAS_SIZE]) / static_cast<T>(sum);
}

template <class T> void initialize(T * u, T * delta, int n_units) {
	for (auto i = 0; i < n_units; i++) {
		u[i] = static_cast<T>(0);
		delta[i] = static_cast<T>(0);
	}
}

template <typename Z, typename INPUT> void prepare_z(Z * z, INPUT * input, int start_addr, int end_addr) {
	z[0] = static_cast<Z>(0);
	for (auto i = start_addr; i < end_addr; i++)
		z[i + BIAS_SIZE - start_addr] = static_cast<Z>(input[i]) / 255.0;
}

template <class T> void set_ans(T * d, unsigned char ans) {
	for (auto i = BIAS_SIZE; i < MNIST_NUMBER_OF_OUTPUT_CLASS + BIAS_SIZE; i++)
		d[i] = ((i - BIAS_SIZE) == ans) ? 1 : 0;
}

double gaussDistribution(double mu, double sigma) {
	/// generate random number using Mersenne Twister 32bit ver.
	static std::mt19937 mt(static_cast<unsigned int>(time(NULL)));
	std::normal_distribution<> norm(mu, sigma);
	return norm(mt);
}

template <class T> void initialize_weights(T ** w, int n_units, int n_units_former) {
	for (auto j = 0; j < n_units; j++)
		for (auto i = 0; i < n_units_former; i++)
			w[j][i] = (i == 0) ? static_cast<T>(0) : static_cast<T>(gaussDistribution(0, 0.01));
}

template <class T> bool get_max_prob_val(T * y, T * d) {
	int max_prob_val = 0;
	T tmp_max_val = static_cast<T>(0.0);
	for (auto i = BIAS_SIZE; i < MNIST_NUMBER_OF_OUTPUT_CLASS + BIAS_SIZE; i++)
		if (y[i] > tmp_max_val) {
			max_prob_val = i;
			tmp_max_val = y[i];
		}

	return (d[max_prob_val] == 1);
}

int main(void)
{
	std::ifstream train_src("../data/train-images-idx3-ubyte", std::ios::in | std::ios::binary);
	std::ifstream train_label("../data/train-labels-idx1-ubyte", std::ios::in | std::ios::binary);
	std::ifstream test_src("../data/t10k-images-idx3-ubyte", std::ios::in | std::ios::binary);
	std::ifstream test_label ("../data/t10k-labels-idx1-ubyte", std::ios::in | std::ios::binary);

	if (!train_src || !train_label || !test_src || !test_label) {
		std::cout << "[ERROR] Cannnot Open File" << std::endl;
		return -1;
	}

	img_info * train_img_info = new img_info();
	img_info * test_img_info = new img_info();
	
	parse_idx(&train_src, &train_label, train_img_info);
	parse_idx(&test_src, &test_label, test_img_info);

	char * src_buf = new char[train_img_info->get_size() * train_img_info->get_n_img()];
	train_src.read(src_buf, train_img_info->get_size() * train_img_info->get_n_img());
	unsigned char* uchar_src_buf = reinterpret_cast<unsigned char*>(src_buf);

	char * label_buf = new char[train_img_info->get_n_img()];
	train_label.read(label_buf, train_img_info->get_n_img());
	unsigned char* uchar_label_buf = reinterpret_cast<unsigned char*>(label_buf);

	// input pixels
	double* z = new double[train_img_info->get_size() + BIAS_SIZE];
	for (int i = 0; i < train_img_info->get_size() + BIAS_SIZE; i++)
		z[i] = i == 0 ? 1.0 : 0.0;

	// hidden layer
	double** w_2 = new double*[HIDDEN_LAYER_UNIT + BIAS_SIZE];
	for (int j = 0; j < HIDDEN_LAYER_UNIT + BIAS_SIZE; j++)
		w_2[j] = new double[train_img_info->get_size() + BIAS_SIZE];
	initialize_weights(w_2, HIDDEN_LAYER_UNIT + BIAS_SIZE, train_img_info->get_size() + BIAS_SIZE);
	double* u_2 = new double[HIDDEN_LAYER_UNIT + BIAS_SIZE];
	double* delta_2 = new double[HIDDEN_LAYER_UNIT + BIAS_SIZE];

	// output layer
	double** w_3 = new double*[MNIST_NUMBER_OF_OUTPUT_CLASS + BIAS_SIZE];
	for (int k = 0; k < MNIST_NUMBER_OF_OUTPUT_CLASS + BIAS_SIZE; k++)
		w_3[k] = new double[HIDDEN_LAYER_UNIT + BIAS_SIZE];
	initialize_weights(w_3, MNIST_NUMBER_OF_OUTPUT_CLASS + BIAS_SIZE, HIDDEN_LAYER_UNIT + BIAS_SIZE);
	double* u_3 = new double[HIDDEN_LAYER_UNIT + BIAS_SIZE];
	double* delta_3 = new double[MNIST_NUMBER_OF_OUTPUT_CLASS + BIAS_SIZE];
	double* y = new double[MNIST_NUMBER_OF_OUTPUT_CLASS + BIAS_SIZE];

	// answer
	double* d = new double[MNIST_NUMBER_OF_OUTPUT_CLASS + BIAS_SIZE];

	int count = 0;
	int n_epoch = 100;
	int epoch_count = 0;
	while (epoch_count < n_epoch)
	{
		while (count < train_img_info->get_n_img())
		{
			initialize(u_2, delta_2, HIDDEN_LAYER_UNIT + BIAS_SIZE);
			initialize(u_3, delta_3, MNIST_NUMBER_OF_OUTPUT_CLASS + BIAS_SIZE);
			prepare_z(z, uchar_src_buf, count * train_img_info->get_size(), (count + 1) * train_img_info->get_size());
			set_ans(d, uchar_label_buf[count]);

			// forward propagation
			for (int j = 0; j < HIDDEN_LAYER_UNIT + BIAS_SIZE; j++) {
				if (j == 0) 
					u_2[j] = 1.0;
				else
					for (int i = 0; i < train_img_info->get_size() + BIAS_SIZE; i++)
						u_2[j] += z[i] * w_2[j][i];
			}
			for (int k = BIAS_SIZE; k < MNIST_NUMBER_OF_OUTPUT_CLASS + BIAS_SIZE; k++)
				if (k == 0) 
					u_3[0] = 1.0;
				else
					for (int j = 0; j < HIDDEN_LAYER_UNIT + BIAS_SIZE; j++)
						u_3[k] += ReLU(u_2[j]) * w_3[k][j];

			// calculate output
			soft_max(u_3, y, MNIST_NUMBER_OF_OUTPUT_CLASS);

			// calculate gradients
			for (int k = BIAS_SIZE; k < MNIST_NUMBER_OF_OUTPUT_CLASS + BIAS_SIZE; k++)
				delta_3[k] = y[k] - d[k];
			for (int j = 0; j < HIDDEN_LAYER_UNIT + BIAS_SIZE; j++)
				for (int k = BIAS_SIZE; k < MNIST_NUMBER_OF_OUTPUT_CLASS + BIAS_SIZE; k++)
					delta_2[j] += u_2[j] > 0 ? delta_3[k] * w_3[k][j] : 0;

			// update paramaters
			for (int j = 0; j < HIDDEN_LAYER_UNIT + BIAS_SIZE; j++)
				for (int k = BIAS_SIZE; k < MNIST_NUMBER_OF_OUTPUT_CLASS + BIAS_SIZE; k++)
					w_3[k][j] -= (j == 0) ?
						LEARNING_RATE_BIAS * delta_3[k] :
						LEARNING_RATE_WEIGHT * delta_3[k] * ReLU(u_2[j]);
			for (int i = 0; i < train_img_info->get_size() + BIAS_SIZE; i++)
				for (int j = BIAS_SIZE; j < HIDDEN_LAYER_UNIT + BIAS_SIZE; j++)
					w_2[j][i] -= (i == 0) ?
						LEARNING_RATE_BIAS * delta_2[j] :
						LEARNING_RATE_WEIGHT * delta_2[j] * z[i];

			count++;

			if (count % 10000 == 0)
				test(w_2, w_3, u_2, u_3, d, z, y, delta_2, delta_3, test_img_info, &test_src, &test_label);
		}
		n_epoch++;
		count = 0;
	}

	delete[] src_buf;
	delete[] label_buf;
}

void parse_idx(std::ifstream * const ifs_src, std::ifstream * const ifs_label, img_info * const obj_img_info)
{
	char* buf = new char[SRC_HEADER_OFFSET];

	ifs_src->read(buf, SRC_HEADER_OFFSET);
	unsigned char* uchar_buf = reinterpret_cast<unsigned char*>(buf);
	
	int n_img = 0;
	for (int i = 0; i < 4; i++)
		n_img |= static_cast<int>(uchar_buf[IDX_OFFSET_NUMBER_OF_IMAGES + i] << (8 * (3 - i)));
	
	int width = 0;
	for (int i = 0; i < 4; i++)
		width |= static_cast<int>(uchar_buf[IDX_OFFSET_NUMBER_OF_ROWS + i] << (8 * (3 - i)));
	
	int height = 0;
	for (int i = 0; i < 4; i++)
		height |= static_cast<int>(uchar_buf[IDX_OFFSET_NUMBER_OF_COLUMNS + i] << (8 * (3 - i)));

	obj_img_info->set_param(n_img, width, height, MNIST_NUMBER_OF_OUTPUT_CLASS);

	ifs_label->read(buf, LABEL_HEADER_OFFSET);

	delete[] buf;
}

template <typename T> void test(T ** const w_2, T ** const w_3, T * const u_2, T * const u_3, T * const d, T * const z, T * const y, T * const delta_2, T * const delta_3, img_info * const test_img_info, 
	std::ifstream * const ifs_src, std::ifstream * const ifs_label)
{
	char * src_buf = new char[test_img_info->get_size() * test_img_info->get_n_img()];
	ifs_src->read(src_buf, test_img_info->get_size() * test_img_info->get_n_img());
	unsigned char* uchar_src_buf = reinterpret_cast<unsigned char*>(src_buf);

	char * label_buf = new char[test_img_info->get_n_img()];
	ifs_label->read(label_buf, test_img_info->get_n_img());
	unsigned char* uchar_label_buf = reinterpret_cast<unsigned char*>(label_buf);

	int count = 0;
	int n_correct_answer = 0;
	while (count < test_img_info->get_n_img())
	{
		initialize(u_2, delta_2, HIDDEN_LAYER_UNIT + BIAS_SIZE);
		initialize(u_3, delta_3, MNIST_NUMBER_OF_OUTPUT_CLASS + BIAS_SIZE);
		prepare_z(z, uchar_src_buf, count * test_img_info->get_size(), (count + 1) * test_img_info->get_size());
		set_ans(d, uchar_label_buf[count]);

		// forward propagation
		for (int j = 0; j < HIDDEN_LAYER_UNIT + BIAS_SIZE; j++)
			for (int i = 0; i < test_img_info->get_size() + BIAS_SIZE; i++)
				u_2[j] += z[i] * w_2[j][i];

		for (int k = 0; k < MNIST_NUMBER_OF_OUTPUT_CLASS + BIAS_SIZE; k++)
			for (int j = 0; j < HIDDEN_LAYER_UNIT + BIAS_SIZE; j++)
				u_3[k] += ReLU(u_2[j]) * w_3[k][j];

		soft_max(u_3, y, MNIST_NUMBER_OF_OUTPUT_CLASS);

		if (get_max_prob_val(y, d))
			n_correct_answer++;

		count++;
	}

	ifs_src->clear();
	ifs_src->seekg(SRC_HEADER_OFFSET);
	ifs_label->clear();
	ifs_label->seekg(LABEL_HEADER_OFFSET);

	std::cout << "[INFO] Accuracy : " << static_cast<double>(n_correct_answer) / static_cast<double>(count) * 100.0 << std::endl;
}
