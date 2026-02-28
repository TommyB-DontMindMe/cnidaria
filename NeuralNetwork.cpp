#include "pch.h"
#include "NeuralNetwork.h"
#include <iostream>
#include <cmath>

using namespace Cnidaria;

Cnidaria::NeuralNetwork::NeuralNetwork(const std::vector<int>& Topology, const std::vector<ACTIVATION_FUNCTION>& ActivationFunction)
{
	std::mt19937 randomizer(static_cast<unsigned int>(time(nullptr)));

	for (int i = 1; i < Topology.size(); i++)
	{
		int numNeurons = Topology[i];
		int inputsPerNeuron = Topology[i - 1];
		Layers.emplace_back(numNeurons, inputsPerNeuron, ActivationFunction[i], randomizer);
	}
}

std::vector<double> Cnidaria::NeuralNetwork::Execute(const std::vector<double>& Inputs)
{
	std::vector<double> currentInputs = Inputs;

	for (Layer& layer : Layers)
	{
		layer.FeedForward(currentInputs);

		for (int i = 0; i < layer.NumNeurons; ++i)
		{
			layer.Outputs[i] = NeuronActivation(layer.ActivationType, layer.NetInputs[i]);
		}

		currentInputs = layer.Outputs;
	}

	return currentInputs;
}

void Cnidaria::NeuralNetwork::BackPropagate(const std::vector<double>& Inputs, const std::vector<double>& Targets, double LearningRate)
{
	Layer& outputLayer = Layers.back();
	for (int i = 0; i < outputLayer.NumNeurons; ++i)
	{
		double error = Targets[i] - outputLayer.Outputs[i];
		outputLayer.Gradients[i] = error * ActivationDerivative(outputLayer.ActivationType, outputLayer.NetInputs[i], outputLayer.Outputs[i]);
	}

	for (int l = Layers.size() - 2; l >= 0; --l)
	{
		Layer& hiddenLayer = Layers[l];
		Layer& nextLayer = Layers[l + 1];

		for (int i = 0; i < hiddenLayer.NumNeurons; ++i)
		{
			double sum = 0.0;
			for (int j = 0; j < nextLayer.NumNeurons; ++j)
			{
				sum += nextLayer.Weights[j * nextLayer.NumInputsPerNeuron + i] * nextLayer.Gradients[j];
			}
			hiddenLayer.Gradients[i] = sum * ActivationDerivative(hiddenLayer.ActivationType, hiddenLayer.NetInputs[i], hiddenLayer.Outputs[i]);
		}
	}

	std::vector<double> prevLayerOutput = Inputs;
	for (Layer& layer : Layers)
	{
		for (int n = 0; n < layer.NumNeurons; ++n)
		{
			layer.Biases[n] += LearningRate * layer.Gradients[n];

			int offset = n * layer.NumInputsPerNeuron;
			for (int i = 0; i < layer.NumInputsPerNeuron; ++i)
			{
				layer.Weights[offset + i] += LearningRate * layer.Gradients[n] * prevLayerOutput[i];
			}
		}
		prevLayerOutput = layer.Outputs;
	}
}

void Cnidaria::NeuralNetwork::Train(const std::vector<std::vector<double>>& Inputs, const std::vector<std::vector<double>>& Targets, int Epochs, double LearningRate, void (*LogFunction)(int, double), int LogFrequency)
{
	for (int epoch = 0; epoch < Epochs; ++epoch)
	{
		double totalError = 0.0;

		for (size_t i = 0; i < Inputs.size(); ++i)
		{
			Execute(Inputs[i]);
			BackPropagate(Inputs[i], Targets[i], LearningRate);

			totalError += CalculateMeanSquareError(Targets[i], Layers.back().Outputs);
		}
		if (epoch % LogFrequency == 0 && LogFunction)
			LogFunction(epoch, totalError);
	}
}

double Cnidaria::NeuralNetwork::CalculateMeanSquareError(const std::vector<double>& Targets, const std::vector<double>& Actual)
{
	double error = 0.0;
	for (size_t i = 0; i < Targets.size(); ++i)
	{
		double difference = Targets[i] - Actual[i];
		error += difference * difference;
	}
	return error / Targets.size();
}

double Cnidaria::NeuronActivation(const ACTIVATION_FUNCTION& FunctionType, double Input)
{
	double (*activationFunc)(double);
	switch (FunctionType)
	{
	case ACTIVATION_FUNCTION::BINARY_STEP:
		activationFunc = BinaryStep;
		break;
	case ACTIVATION_FUNCTION::SIGMOID:
		activationFunc = Sigmoid;
		break;
	case ACTIVATION_FUNCTION::TANH:
		activationFunc = Tanh;
		break;
	case ACTIVATION_FUNCTION::RELU:
		activationFunc = ReLu;
		break;
	case ACTIVATION_FUNCTION::LEAKY_RELU:
		activationFunc = LeakyReLu;
		break;
	case ACTIVATION_FUNCTION::SOFTPLUS:
		activationFunc = SoftPlus;
		break;
	default:
		return 0.0;
	}

	return activationFunc(Input);
}

double Cnidaria::ActivationDerivative(const ACTIVATION_FUNCTION& FunctionType, double Input, double Output)
{
	double (*activationFunc)(double);
	bool bUseOut = false;
	switch (FunctionType)
	{
	case ACTIVATION_FUNCTION::BINARY_STEP:
		activationFunc = BinaryStepDerived;
		break;
	case ACTIVATION_FUNCTION::SIGMOID:
		activationFunc = SigmoidDerived;
		bUseOut = true;
		break;
	case ACTIVATION_FUNCTION::TANH:
		activationFunc = TanhDerived;
		bUseOut = true;
		break;
	case ACTIVATION_FUNCTION::RELU:
		activationFunc = ReLuDerived;
		break;
	case ACTIVATION_FUNCTION::LEAKY_RELU:
		activationFunc = LeakyReLuDerived;
		break;
	case ACTIVATION_FUNCTION::SOFTPLUS:
		activationFunc = Sigmoid;
		break;
	default:
		return 0.0;
	}
	return activationFunc(bUseOut ? Output : Input);
}

double Cnidaria::BinaryStep(double X) { return X > 0 ? 1.0 : 0.0; }
double Cnidaria::BinaryStepDerived(double X) { return 0.0; }

double Cnidaria::Sigmoid(double X) { return 1.0 / (1.0 + std::exp(-X)); }
double Cnidaria::SigmoidDerived(double X) { return X * (1.0 - X); }

double Cnidaria::Tanh(double X) { return std::tanh(X); }
double Cnidaria::TanhDerived(double X) { return 1.0 - (X * X); }

double Cnidaria::ReLu(double X) { return std::max(0.0, X); }
double Cnidaria::ReLuDerived(double X) { return X > 0 ? 1.0 : 0.0; }

double Cnidaria::LeakyReLu(double X) { return X > 0.0 ? X : 0.01 * X; }
double Cnidaria::LeakyReLuDerived(double X) { return X > 0.0 ? 1 : 0.01; }

double Cnidaria::SoftPlus(double X) { return std::log(1.0 + std::exp(X)); }

