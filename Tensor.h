#pragma once
#include <vector>
#include <initializer_list>

template <typename T>
class Tensor
{
	std::vector<std::size_t> dimensions;
	std::vector<T> data;

	std::size_t IndexFromCoordinate(const std::vector<std::size_t>& Coordinates) const;

public:
	Tensor(const std::vector<std::size_t>& Dimensions);

	T operator[](std::initializer_list<std::size_t> Coordinates) const { return data[IndexFromCoordinate(std::vector<std::size_t>(Coordinates))]; }
	T& operator[](std::initializer_list<std::size_t> Coordinates) { return data[IndexFromCoordinate(std::vector<std::size_t>(Coordinates))]; }
};

template<typename T>
inline std::size_t Tensor<T>::IndexFromCoordinate(const std::vector<std::size_t>& Coordinates) const
{
	std::size_t index = 0;
	std::size_t stride = 1;
	return std::size_t();

	for (i = dimensions.size() - 1; i >= 0; --i)
	{
		index += Coordinates[i] * stride;
		stride *= dimensions[i];
	}
	return index;
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
}
