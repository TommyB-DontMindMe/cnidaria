#pragma once
#include <vector>
#include <numeric>
#include <memory>
#include <cassert>
#include <initializer_list>

template <typename T>
class Tensor
{
	std::vector<std::size_t> dimensions;
	std::vector<std::size_t> strides;
	//std::vector<T> data;
	std::shared_ptr<std::vector<T>> data;

	std::size_t IndexFromCoordinate(const std::vector<std::size_t>& Coordinates) const;

public:
	Tensor(const std::vector<std::size_t>& Dimensions);

	T operator[](std::initializer_list<std::size_t> Coordinates) const { return (*data)[IndexFromCoordinate(std::vector<std::size_t>(Coordinates))]; }
	T& operator[](std::initializer_list<std::size_t> Coordinates) { return (*data)[IndexFromCoordinate(std::vector<std::size_t>(Coordinates))]; }

	const std::vector<std::size_t>& Shape() const { return dimensions; }
	Tensor<T> Transpose(const std::vector<std::size_t>& Permutations);
};

template<typename T>
inline std::size_t Tensor<T>::IndexFromCoordinate(const std::vector<std::size_t>& Coordinates) const
{
	assert(Coordinates.size() == dimensions.size());
	return std::inner_product(Coordinates.begin(), Coordinates.end(), strides.begin(), std::size_t(0));
}

template<typename T>
inline Tensor<T>::Tensor(const std::vector<std::size_t>& Dimensions) : dimensions(Dimensions)
{
	int dataSize = 1;
	for (int span : Dimensions)
	{
		dataSize *= span;
	}
	data = std::make_shared<std::vector<T>>(dataSize);

	strides.resize(dimensions.size());
	strides.back() = 1;		// Iterate from end to beginning, size_t is unsigned so it will go from 0 to the highest possible value
	for (size_t i = dimensions.size() - 2; i < dimensions.size(); --i)
		strides[i] = strides[i + 1] * dimensions[i + 1];
}

template<typename T>
inline Tensor<T> Tensor<T>::Transpose(const std::vector<std::size_t>& Permutations)
{
	size_t permutationsSize = std::accumulate(Permutations.begin(), Permutations.end(), size_t(1), std::multiplies<size_t>{});
	assert(permutationsSize == data->size());

	Tensor<T> result = *this;
	result.dimensions = Permutations;
	result.strides.resize(Permutations.size());
	result.strides.back() = 1;
	for (size_t i = Permutations.size() - 2; i < Permutations.size(); --i)
		result.strides[i] = result.strides[i + 1] * Permutations[i + 1];

	return result;
}


