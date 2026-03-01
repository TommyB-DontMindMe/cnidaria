#pragma once

#include <vector>
#include <random>

namespace Cnidaria
{
	enum ACTIVATION_FUNCTION
	{
		BINARY_STEP, SIGMOID, TANH, RELU, LEAKY_RELU, SOFTPLUS
	};

	class Layer
	{
	public:
		int NumNeurons;
		int NumInputsPerNeuron;	//Generally equals NumNeurons of previous Layer
		ACTIVATION_FUNCTION ActivationType;

		std::vector<double> Outputs;
		std::vector<double> Biases;
		std::vector<double> Gradients;
		std::vector<double> NetInputs;

		std::vector<double> Weights;

		Layer(int NeuronCount, int InputCount, ACTIVATION_FUNCTION ActivationType, std::mt19937& RandomnessGenerator);
		Layer(ACTIVATION_FUNCTION ActivationType, const std::vector<double>& Biases, const std::vector<double>& Weights);

		void FeedForward(const std::vector<double>& Inputs);
	};
}