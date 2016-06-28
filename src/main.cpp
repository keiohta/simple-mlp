/**
 * @file main.cpp
 * @brief simple mlp
 * @author k.ohta
 * @date 2016/06/23
 * @editor T.abe
 * @date updated on 2016/06/26 Sun
 */

#include <memory>
#include <iostream>
#include <fstream>
#include <random>
#include <ctime>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <numeric>

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
public:
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
private:
	int n_img;
	int size;
	int n_output_class;
};

void parse_idx(std::ifstream * const ifs_src, std::ifstream * const ifs_label, img_info * const obj_img_info);
template <typename T> void test(
	std::vector<std::vector<T>> &w_2,
	std::vector<std::vector<T>> &w_3,
	std::vector<T> &u_2,
	std::vector<T> &u_3,
	std::vector<T> &d,
	std::vector<T> &z,
	std::vector<T> &y,
	std::vector<T> &delta_2,
	std::vector<T> &delta_3,
	img_info * test_img_info,
	std::ifstream * const ifs_src,
	std::ifstream * const ifs_label
	);

template <class T> T ReLU(T val) {
	return val > static_cast<T>(0) ? val : static_cast<T>(0);
}

template<class T> std::vector<T> ReLU_vec(std::vector<T> &val) {
	std::vector<T> res;
	for (T x : val)
		res.push_back(x > static_cast<T>(0) ? x : static_cast<T>(0));
	return res;
}

template <class T> void soft_max(std::vector<T> &u, std::vector<T> &y, int n_output_class) {
	T sum = static_cast<T>(0);
	for (auto k = 0; k < n_output_class; k++)
		sum += std::exp(u[k + BIAS_SIZE]);
	for (auto k = 0; k < n_output_class; k++)
		y[k + BIAS_SIZE] = std::exp(u[k + BIAS_SIZE]) / static_cast<T>(sum);
}

template <class T> void initialize(std::vector<T> &u, std::vector<T> &delta, int n_units) {
	for (auto i = 0; i < n_units; i++) {
		u[i] = static_cast<T>(0);
		delta[i] = static_cast<T>(0);
	}
}

template <typename Z, typename INPUT> void prepare_z(std::vector<Z> &z, std::vector<INPUT> &input, int start_addr, int end_addr) {
	z[0] = static_cast<Z>(0);
	for (auto i = start_addr; i < end_addr; i++)
		z[i + BIAS_SIZE - start_addr] = static_cast<Z>(input[i]) / 255.0;
}

template <class T> void set_ans(std::vector<T> &d, unsigned char ans) {
	for (auto i = BIAS_SIZE; i < MNIST_NUMBER_OF_OUTPUT_CLASS + BIAS_SIZE; i++)
		d[i] = ((i - BIAS_SIZE) == ans) ? 1 : 0;
}

double gaussDistribution(double mu, double sigma) {
	//! generate random number using Mersenne Twister 32bit ver.
	static std::mt19937 mt(static_cast<unsigned int>(time(NULL)));
	std::normal_distribution<> norm(mu, sigma);
	return norm(mt);
}

template <class T> void initialize_weights(std::vector<std::vector<T>> &w, int n_units, int n_units_former) {
	for (auto j = 0; j < n_units; j++)
		for (auto i = 0; i < n_units_former; i++)
			w[j][i] = (i == 0) ? static_cast<T>(0) : static_cast<T>(gaussDistribution(0, 0.01));
}

template <class T> bool get_max_prob_val(std::vector<T> &y, std::vector<T> &d) {
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
	try {
		std::ifstream train_src("../data/train-images-idx3-ubyte", std::ios::in | std::ios::binary);
		std::ifstream train_label("../data/train-labels-idx1-ubyte", std::ios::in | std::ios::binary);
		std::ifstream test_src("../data/t10k-images-idx3-ubyte", std::ios::in | std::ios::binary);
		std::ifstream test_label("../data/t10k-labels-idx1-ubyte", std::ios::in | std::ios::binary);

		//! Error return if files not exist
		if (!train_src || !train_label || !test_src || !test_label) {
			throw "[ERROR] Cannnot Open File";
		}

		img_info * train_img_info = new img_info();
		img_info * test_img_info = new img_info();

		parse_idx(&train_src, &train_label, train_img_info);
		parse_idx(&test_src, &test_label, test_img_info);

		std::vector<char> src_buf(train_img_info->get_size() * train_img_info->get_n_img());
		train_src.read(&src_buf.front(), train_img_info->get_size() * train_img_info->get_n_img());
		std::vector<unsigned char> uchar_src_buf(src_buf.begin(), src_buf.end());

		std::vector<char> label_buf(train_img_info->get_n_img());
		train_label.read(&label_buf.front(), train_img_info->get_n_img());
		std::vector<unsigned char> uchar_label_buf(label_buf.begin(), label_buf.end());

		//! input pixels
		std::vector <double> z(train_img_info->get_size() + BIAS_SIZE);
		std::fill(z.begin(), z.end(), 0.0);
		z[0] = 1.0;

		//! hidden layer
		std::vector<std::vector<double>> w_2(HIDDEN_LAYER_UNIT + BIAS_SIZE, std::vector<double>(train_img_info->get_size() + BIAS_SIZE));
		initialize_weights(w_2, HIDDEN_LAYER_UNIT + BIAS_SIZE, train_img_info->get_size() + BIAS_SIZE);
		std::vector<double> u_2(HIDDEN_LAYER_UNIT + BIAS_SIZE);
		std::vector<double> delta_2(HIDDEN_LAYER_UNIT + BIAS_SIZE);

		//! output layer
		std::vector<std::vector<double>> w_3(MNIST_NUMBER_OF_OUTPUT_CLASS + BIAS_SIZE, std::vector<double>(HIDDEN_LAYER_UNIT + BIAS_SIZE));
		initialize_weights(w_3, MNIST_NUMBER_OF_OUTPUT_CLASS + BIAS_SIZE, HIDDEN_LAYER_UNIT + BIAS_SIZE);
		std::vector<double> u_3(HIDDEN_LAYER_UNIT + BIAS_SIZE);
		std::vector<double> delta_3(MNIST_NUMBER_OF_OUTPUT_CLASS + BIAS_SIZE);
		std::vector<double> y(MNIST_NUMBER_OF_OUTPUT_CLASS + BIAS_SIZE);

		//! answer
		std::vector<double> d(MNIST_NUMBER_OF_OUTPUT_CLASS + BIAS_SIZE);

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

				//! forward propagation
				for (int j = 0; j < HIDDEN_LAYER_UNIT + BIAS_SIZE; j++) {
					if (j == 0)
						u_2[j] = 1.0;
					else
						/*
						for (int i = 0; i < train_img_info->get_size() + BIAS_SIZE; i++)
						u_2[j] += z[i] * w_2[j][i];
						*/
						u_2[j] = std::inner_product(z.begin(), z.end(), w_2[j].begin(), 0.0);
				}
				for (int k = BIAS_SIZE; k < MNIST_NUMBER_OF_OUTPUT_CLASS + BIAS_SIZE; k++) {
					if (k == 0)
						u_3[k] = 1.0;
					else {
						/*
						for (int j = 0; j < HIDDEN_LAYER_UNIT + BIAS_SIZE; j++)
						u_3[k] += ReLU(u_2[j]) * w_3[k][j];
						*/
						u_2 = ReLU_vec(u_2);
						u_3[k] = std::inner_product(u_2.begin(), u_2.end(), w_3[k].begin(), 0.0);
					}
				}

				//! calculate output
				soft_max(u_3, y, MNIST_NUMBER_OF_OUTPUT_CLASS);

				//! calculate gradients
				std::transform(y.begin(), y.end(), d.begin(), delta_3.begin(), std::minus<double>());
				for (int j = 0; j < HIDDEN_LAYER_UNIT + BIAS_SIZE; j++) {
					for (int k = BIAS_SIZE; k < MNIST_NUMBER_OF_OUTPUT_CLASS + BIAS_SIZE; k++)
						delta_2[j] += u_2[j] > 0 ? delta_3[k] * w_3[k][j] : 0;
				}
				//! update paramaters
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

	}
	catch (char* str) {
		std::cout << str << std::endl;
		return -1;
	}

	return 0;
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

template <typename T> void test(
	std::vector<std::vector<T>> &w_2,
	std::vector<std::vector<T>> &w_3,
	std::vector<T> &u_2,
	std::vector<T> &u_3,
	std::vector<T> &d,
	std::vector<T> &z,
	std::vector<T> &y,
	std::vector<T> &delta_2,
	std::vector<T> &delta_3,
	img_info * const test_img_info,
	std::ifstream * const ifs_src,
	std::ifstream * const ifs_label
	)
{
	std::vector<char> src_buf(test_img_info->get_size() * test_img_info->get_n_img());
	ifs_src->read(&src_buf.front(), test_img_info->get_size() * test_img_info->get_n_img());
	std::vector<unsigned char> uchar_src_buf(src_buf.begin(), src_buf.end());

	std::vector<char> label_buf(test_img_info->get_n_img());
	ifs_label->read(&label_buf.front(), test_img_info->get_n_img());
	std::vector<unsigned char> uchar_label_buf(label_buf.begin(), label_buf.end());

	int count = 0;
	int n_correct_answer = 0;
	while (count < test_img_info->get_n_img())
	{
		initialize(u_2, delta_2, HIDDEN_LAYER_UNIT + BIAS_SIZE);
		initialize(u_3, delta_3, MNIST_NUMBER_OF_OUTPUT_CLASS + BIAS_SIZE);
		prepare_z(z, uchar_src_buf, count * test_img_info->get_size(), (count + 1) * test_img_info->get_size());
		set_ans(d, uchar_label_buf[count]);

		//! forward propagation
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
