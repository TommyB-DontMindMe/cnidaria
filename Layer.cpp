#include "pch.h"
#include "Layer.h"

using namespace Cnidaria;

Layer::Layer(int NeuronCount, int InputCount, ACTIVATION_FUNCTION ActivationType, std::mt19937& RandomnessGenerator) : NumNeurons(NeuronCount), NumInputsPerNeuron(InputCount), ActivationType(ActivationType)
{
	std::uniform_real_distribution<double> distribution(-1.0, 1.0);

	Outputs.resize(NumNeurons, 0.0);
	Gradients.resize(NumNeurons, 0.0);
	NetInputs.resize(NumNeurons, 0.0);

	Biases.resize(NumNeurons);
	for (double& b : Biases)
	{
		b = distribution(RandomnessGenerator);
	}
	Weights.resize(NumNeurons * NumInputsPerNeuron);
	for (double& w : Weights)
	{
		w = distribution(RandomnessGenerator);
	}
}

void Layer::FeedForward(const std::vector<double>& Inputs)
{
	for (int n = 0; n < NumNeurons; n++)
	{
		double sum = Biases[n];
		int weightOffset = n * NumInputsPerNeuron;
		for (int i = 0; i < NumInputsPerNeuron; i++)
		{
			sum += Inputs[i] * Weights[weightOffset + i];
		}

		NetInputs[n] = sum;
	}
}
