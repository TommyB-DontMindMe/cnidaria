#pragma once
#include <vector>
#include <numeric>
#include <initializer_list>

template <typename T>
class Tensor
{
	std::vector<std::size_t> dimensions;
	std::vector<std::size_t> strides;
	std::vector<T> data;

	std::size_t IndexFromCoordinate(const std::vector<std::size_t>& Coordinates) const;

public:
	Tensor(const std::vector<std::size_t>& Dimensions);

	T operator[](std::initializer_list<std::size_t> Coordinates) const { return data[IndexFromCoordinate(std::vector<std::size_t>(Coordinates))]; }
	T& operator[](std::initializer_list<std::size_t> Coordinates) { return data[IndexFromCoordinate(std::vector<std::size_t>(Coordinates))]; }

	const std::vector<std::size_t>& Shape() const { return dimensions; }
	Tensor<T> Transpose(const std::vector<std::size_t>& Permutations);
};

template<typename T>
inline std::size_t Tensor<T>::IndexFromCoordinate(const std::vector<std::size_t>& Coordinates) const
{
	assert(Coordinates.size() == dimensions.size());
	return std::inner_product(Coordinates.begin(), Coordinates.end(), strides.begin(), strides.end(), std::size_t(0));

}

template<typename T>
inline Tensor<T>::Tensor(const std::vector<std::size_t>& Dimensions)
{
	int dataSize = 1;
	for (int span : Dimensions)
	{
		dataSize *= span;
	}
	data.resize(dataSize);

	dimensions = Dimensions;
	strides.resize(dimensions.size());
	strides.back() = 1;
	for (size_t i = dimensions.size() - 2; i >= 0; --i)
		strides[i] = strides[i + 1] * dimensions[i + 1];
}

template<typename T>
inline Tensor<T> Tensor<T>::Transpose(const std::vector<std::size_t>& Permutations)
{
	assert(Permutations.size() == dimensions.size());

	Tensor<T> result = *this;
	for (size_t i = 0; i < Permutations.size(); ++i)
	{

	}
	return result;
}


