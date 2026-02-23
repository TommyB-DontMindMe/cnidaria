#pragma once

#include "Layer.h"

class Tensor;

namespace Cnidaria
{
	class NeuralNetwork
	{
	public:
		std::vector<Layer> Layers;

		// Topology specify the number of neurons in each layer
		// ActivationFunction sets the activation function for each layer
		// The first ActivationFunction is not used as it corresponds to the Input layer
		NeuralNetwork(const std::vector<int>& Topology, const std::vector<ACTIVATION_FUNCTION>& ActivationFunction);

		std::vector<double> Execute(const std::vector<double>& Inputs);

		void BackPropagate(const std::vector<double>& Inputs, const std::vector<double>& Targets, double LearningRate);

		void Train(const std::vector<std::vector<double>>& Inputs, const std::vector<std::vector<double>>& Targets, int Epochs, double LearningRate);

		double CalculateMeanSquareError(const std::vector<double>& Targets, const std::vector<double>& Actual);
	};

	double NeuronActivation(const ACTIVATION_FUNCTION& FunctionType, double Input);
	double ActivationDerivative(const ACTIVATION_FUNCTION& FunctionType, double NetInput, double Output);


	double BinaryStep(double X);
	double BinaryStepDerived(double X);

	double Sigmoid(double X);
	double SigmoidDerived(double X);

	double Tanh(double X);
	double TanhDerived(double X);

	double ReLu(double X);
	double ReLuDerived(double X);

	double LeakyReLu(double X);
	double LeakyReLuDerived(double X);

	double SoftPlus(double X);
	// SoftPlus derived equal Sigmoid and uses that instead
}
