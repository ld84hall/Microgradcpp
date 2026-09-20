#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_set>
#include <list>
#include <algorithm>


using namespace std;

class Value;
shared_ptr<Value> operator+(const shared_ptr<Value>& lhs, const shared_ptr<Value>& rhs);
shared_ptr<Value> operator*(const shared_ptr<Value>& lhs, const shared_ptr<Value>& rhs);
shared_ptr<Value> operator-(const shared_ptr<Value>& lhs, const shared_ptr<Value>& rhs);
shared_ptr<Value> custom_tanh(const shared_ptr<Value>& number);
shared_ptr<Value> custom_pow(const shared_ptr<Value>& y, double x);
shared_ptr<Value> custom_sigmoid(const shared_ptr<Value>& number);
shared_ptr<Value> custom_log(const shared_ptr<Value>& number);


class Value : public enable_shared_from_this<Value> {
public:
	double data;
	double grad = 0.0;
	vector<shared_ptr<Value>> _children;
	function<void()>_backward_fn = [](){};

	Value(double val, vector<shared_ptr<Value>> children = {})
	  : data(val), _children(children) {}

	shared_ptr<Value> tanh() {
		return custom_tanh(shared_from_this());
	};

	shared_ptr<Value> sigmoid() {
		return custom_sigmoid(shared_from_this());
	};
};

shared_ptr<Value> operator+(const shared_ptr<Value>& lhs, const shared_ptr<Value>& rhs) {
	vector<shared_ptr<Value>> children_list = {lhs, rhs};
	shared_ptr<Value> out =  make_shared<Value>(lhs->data + rhs->data, children_list);

	out->_backward_fn = [=]() {
		lhs->grad += out->grad;
		rhs->grad += out->grad;
	};

	return out;
};

shared_ptr<Value> operator*(const shared_ptr<Value>& lhs, const shared_ptr<Value>& rhs) {
	vector<shared_ptr<Value>> children_list = {lhs, rhs};
	shared_ptr<Value> out = make_shared<Value>(lhs->data * rhs->data, children_list);

	out->_backward_fn = [=]() {
		lhs->grad += (rhs->data)*(out->grad);
		rhs->grad += (lhs->data)*(out->grad);
	};

	return out;	
};

shared_ptr<Value> operator-(const shared_ptr<Value>& lhs, const shared_ptr<Value>& rhs) {
	vector<shared_ptr<Value>> children_list = {lhs, rhs};
	shared_ptr<Value> out = make_shared<Value>(lhs->data - rhs->data, children_list);

	out->_backward_fn = [=]() {
		lhs->grad += out->grad;
		rhs->grad += out->grad*-1;
	};

	return out;

};

shared_ptr<Value> operator/(const shared_ptr<Value>& lhs, const shared_ptr<Value>& rhs) {
	return (lhs)*(custom_pow((rhs), -1));
};

shared_ptr<Value> custom_tanh(const shared_ptr<Value>& number) {
	double x = number->data;
	double t = (exp(2*number->data) - 1)/(exp(2*number->data) + 1);

	vector<shared_ptr<Value>> children_list = {number,};
	shared_ptr<Value> out = make_shared<Value>(t, children_list);

	out->_backward_fn = [=]() {
		number->grad += out->grad * (1.0 - (t*t));
	};
	return out;
};

shared_ptr<Value> custom_sigmoid(const shared_ptr<Value>& number){
	double x = number->data;
	double sigma = 1/(1+exp(-x));

	vector<shared_ptr<Value>> children_list = {number,};
	shared_ptr<Value> out = make_shared<Value>(sigma, children_list);

	out->_backward_fn = [=](){
		number->grad += out->grad * sigma * (1-sigma);
	};

	return out;
};

shared_ptr<Value> custom_log(const shared_ptr<Value>& number) {
	double x = number->data;
	double t = log(x);

	vector<shared_ptr<Value>> children_list = {number, };
	shared_ptr<Value> out = make_shared<Value>(t, children_list);

	out->_backward_fn = [=](){
		number->grad += out->grad / x;
	};

	return out;

}

shared_ptr<Value> custom_pow(const shared_ptr<Value>& y, double x) {
	vector<shared_ptr<Value>> children_list = {y, };
	shared_ptr<Value> out = make_shared<Value>(pow(y->data,x), children_list);

	out->_backward_fn = [=](){
		y->grad += x * pow(y->data,(x-1)) * out->grad;
	};

	return out;
};

shared_ptr<Value> operator+(double lhs, const shared_ptr<Value>& rhs){
	shared_ptr<Value>newlhs = make_shared<Value>(lhs);
	return newlhs + rhs;
}

shared_ptr<Value> operator+(const shared_ptr<Value>& lhs, double rhs){
	shared_ptr<Value>newrhs = make_shared<Value>(rhs);
	return newrhs + lhs;
}

shared_ptr<Value> operator*(double lhs, const shared_ptr<Value>& rhs){
	shared_ptr<Value>newlhs = make_shared<Value>(lhs);
	return newlhs * rhs;
}

shared_ptr<Value> operator*(const shared_ptr<Value>& lhs, double rhs){
	shared_ptr<Value>newrhs = make_shared<Value>(rhs);
	return newrhs * lhs;
}

void backward(const shared_ptr<Value>& x){
	vector<shared_ptr<Value>> topo = {};
	unordered_set<Value*> visited = {};

	function<void(const shared_ptr<Value>&)> build_topo = [&](const shared_ptr<Value>& v) {
		bool found = (visited.find(v.get()) != visited.end());
		if (found == false) {
			visited.insert(v.get());
			for (const shared_ptr<Value>& child : v->_children){
				build_topo(child);
			};
			topo.push_back(v);
		};
};
	
	build_topo(x);
	reverse(topo.begin(), topo.end());
	x->grad = 1.0;
	for (int i=0; i<topo.size(); i++){
		topo[i]->_backward_fn();
	};
};


vector<shared_ptr<Value>> randomlist(int n){
	vector<shared_ptr<Value>> array = {};
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> dis(-1.0, 1.0);
	for (int i=0; i<n; i++)
		array.push_back(make_shared<Value>(dis(gen)));
	return array;
};

double randomnum(){
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> dis(-1.0, 1.0);
	return dis(gen);
};

class Neuron: public enable_shared_from_this<Neuron>{

public:
	vector<shared_ptr<Value>> w = {};
	shared_ptr<Value> b;
	Neuron(int nin) : w(randomlist(nin)), b(make_shared<Value>(randomnum())) {}

	shared_ptr<Value> operator()(const vector<shared_ptr<Value>>& x) const {
		shared_ptr<Value> act = b;
		for (int i=0;i<w.size();i++) {
			shared_ptr<Value> y = x[i];
			act = act + (w[i] * y);
		};
		shared_ptr<Value> out = act->sigmoid();
		return out;

	};

	vector<shared_ptr<Value>> parameters() {
		vector<shared_ptr<Value>> neww = w;
		neww.push_back(b);
		return neww;
	};


};

vector<Neuron> genNeurons(int nin, int nout){
	vector<Neuron> neurons = {};
	for (int i=0; i<nout; i++)
		neurons.push_back(Neuron(nin));
	return neurons;

};

class Layer: public enable_shared_from_this<Layer>{
public:
	vector<Neuron> neurons = {};
	Layer(int nin, int nout) : neurons(genNeurons(nin, nout)) {}

	vector<shared_ptr<Value>> operator()(const vector<shared_ptr<Value>>& x) const {
		vector<shared_ptr<Value>> out = {};
		for (int i = 0; i< neurons.size(); i++)
			out.push_back(neurons[i](x));
		return out;
	};	

	vector<shared_ptr<Value>> parameters(){
		vector<shared_ptr<Value>> params = {};
		for (int i = 0; i < neurons.size(); i++) {
			for (int j = 0; j< neurons[i].parameters().size(); j++){
				params.push_back(neurons[i].parameters()[j]);
			};
		};
		return params;
	};
};

vector<Layer> genLayers(int nin, const vector<int>& nouts){
	vector<Layer> layers = {};
	int current_in = nin;
	for (int i=0; i<nouts.size(); i++){
		layers.push_back(Layer(current_in, nouts[i]));
		current_in = nouts[i];
	};
	return layers;
};


class MLP: public enable_shared_from_this<MLP>{
public:
	vector<Layer> Layers = {};
	MLP(int nin, const vector<int> nouts) : Layers(genLayers(nin, nouts)) {}


	vector<shared_ptr<Value>> operator()(const vector<double>& x) const {
		vector<shared_ptr<Value>> currentvals = {};
		for (int i=0; i<x.size(); i++){
			currentvals.push_back(make_shared<Value>(x[i]));
		};
		for (int i=0; i<Layers.size(); i++)
			currentvals = Layers[i](currentvals);
		
		return currentvals;

	};
	vector<shared_ptr<Value>> parameters(){
		vector<shared_ptr<Value>> params = {};
		for (int i=0; i< Layers.size(); i++)
			for (int j = 0;j <Layers[i].parameters().size(); j++)
					params.push_back(Layers[i].parameters()[j]);
		return params;
	};

};

int main(){
	vector<vector<double>> xs = {
    {0.9752, 0.0076},
    {0.9576, 0.2227},
    {0.8037, 0.4468},
    {0.6558, 0.6083},
    {0.3664, 0.7712},
    {0.0768, 0.8252},
    {-0.2217, 0.8529},
    {-0.4905, 0.7677},
    {-0.7188, 0.6121},
    {-0.8596, 0.4635},
    {-0.9702, 0.2458},
    {-0.9634, -0.0079},
    {-0.9168, -0.1983},
    {-0.7753, -0.4464},
    {-0.6030, -0.5739},
    {0.0248, 0.4924},
    {0.0424, 0.2773},
    {0.1963, 0.0532},
    {0.3442, -0.1083},
    {0.6336, -0.2712},
    {0.9232, -0.3252},
    {1.2217, -0.3529},
    {1.4905, -0.2677},
    {1.7188, -0.1121},
    {1.8596, 0.0365},
    {1.9702, 0.2542},
    {1.9634, 0.5079},
    {1.9168, 0.6983},
    {1.7753, 0.9464},
    {1.6030, 1.0739}
};

vector<double> ys = {
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
};


	MLP n(2, {4,4,1});

	vector<shared_ptr<Value>> ypred = {};
	int size = n.parameters().size();

	vector<double> v(size, 0.0);
	vector<double> m(size, 0.0);
	vector<double> vhat(size, 0.0);
	vector<double> mhat(size, 0.0);

	int i;

	for (i=1; i<100; i++){
		cout << "Doing something\n";
		ypred.clear();
		for (int j = 0; j<xs.size(); j++){
			ypred.push_back(n(xs[j])[0]);
		};
		shared_ptr<Value> loss = make_shared<Value>(0.0);
		for (int j=0; j<ypred.size(); j++){
			shared_ptr<Value> yout1 = make_shared<Value>(ys[j]);
			shared_ptr<Value> yout2 = make_shared<Value>(1-ys[j]);
			shared_ptr<Value> yout4 = 1.0 +  (-1*ypred[j]); 
			loss = loss + (yout1*custom_log(ypred[j]) + (yout2)*custom_log(yout4));
		};
		shared_ptr<Value> num = make_shared<Value>(ypred.size());
		loss = -1*loss * custom_pow(num, -1);
		cout << loss->data << endl; 

		for (int j = 0; j<size; j++){
			n.parameters()[j]->grad = 0.0;
		};


		backward(loss);

		vector<double> grads(size, 0.0);

		for (int j = 0; j<size; j++){
			grads[j] = n.parameters()[j]->grad;
		};

		double vt_prev;
		double vt;
		double B1 = 0.9;
		double B2 = 0.999;
		double e = 1e-8;


		for (int j = 0; j<size; j++){
			v[j] = B2*v[j] + (1-B2)*pow((grads[j]),2);
			vhat[j] = v[j] / (1-pow(B2, i));
			m[j] = B1*m[j] + (1-B1)*(grads[j]);
			mhat[j] = m[j] / (1-pow(B1, i));
		};
		for (int j=0; j<size; j++){
			n.parameters()[j]->data += (-0.1/(pow((vhat[j]), 0.5)+e))*(mhat[j]);
		};

	};
	cout << "\nFinal Network Performance Summary:\n";
	for (size_t j = 0; j < xs.size(); j++) {
		cout << "Input pattern " << j << " Target: " << ys[j] << " -> Prediction: " << n(xs[j])[0]->data << "\n";
	};
	return 0;

};


