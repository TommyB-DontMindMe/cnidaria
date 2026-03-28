#pragma once
#include <vector>
#include <numeric>
#include <memory>
#include <cassert>
#include <initializer_list>

template <typename T>
class Tensor
{
	std::vector<size_t> dimensions;
	std::vector<size_t> strides;
	std::shared_ptr<std::vector<T>> data;
	size_t offset = 0;

	size_t IndexFromCoordinate(const std::vector<size_t>& Coordinates) const;

	Tensor(std::shared_ptr<std::vector<T>> Data, size_t Offset, const std::vector<size_t>& Dimensions, const std::vector<size_t>& Strides) : data(Data), offset(Offset), dimensions(Dimensions), strides(Strides) {}

public:
	Tensor(const std::vector<size_t>& Dimensions);

	T operator[](std::initializer_list<size_t> Coordinates) const { return (*data)[IndexFromCoordinate(std::vector<size_t>(Coordinates))]; }
	T& operator[](std::initializer_list<size_t> Coordinates) { return (*data)[IndexFromCoordinate(std::vector<size_t>(Coordinates))]; }

	const std::vector<std::size_t>& Shape() const { return dimensions; }

	// Retain the data in place but alter how it is accessed

	// Create a new Tensor sharing the same data, can be reshaped
	Tensor<T> View(const std::vector<size_t>& Dimensions) const;

	// Dimension: Index of the dimension to query
	Tensor<T> Slice(size_t Dimension, size_t Start, size_t Length) const;

	// Create a new Tensor sharing the same data, but rearrange strides
	// Permutations: Stride indices in the desired order
	Tensor<T> Transpose(const std::vector<size_t>& Permutations) const;

	bool IsContiguous() const;


};

template<typename T>
inline Tensor<T>::Tensor(const std::vector<size_t>& Dimensions) : dimensions(Dimensions), offset(0)
{
	size_t dataSize = std::accumulate(Dimensions.begin(), Dimensions.end(), size_t(1), std::multiplies<size_t>{});
	data = std::make_shared<std::vector<T>>(dataSize);

	strides.resize(dimensions.size());
	strides.back() = 1;		// Iterate from end to beginning, size_t is unsigned so it will go from 0 to the highest possible value
	for (size_t i = dimensions.size() - 2; i < dimensions.size(); --i)
		strides[i] = strides[i + 1] * dimensions[i + 1];
}

template<typename T>
inline size_t Tensor<T>::IndexFromCoordinate(const std::vector<size_t>& Coordinates) const
{
	assert(Coordinates.size() == dimensions.size());
	return offset + std::inner_product(Coordinates.begin(), Coordinates.end(), strides.begin(), size_t(0));
}

template<typename T>
inline Tensor<T> Tensor<T>::Transpose(const std::vector<size_t>& Permutations) const
{
	assert(Permutations.size() == dimensions.size());

	std::vector<size_t> newDimensions(dimensions.size());
	std::vector<size_t> newStrides(strides.size());

	for (size_t i = 0; 1 < Permutations.size(); ++i)
	{
		newDimensions[i] = dimensions[Permutations[i]];
		newStrides[i] = strides[Permutations[i]];
	}
	return Tensor<T>(data, offset, newDimensions, newStrides);
}

template<typename T>
inline bool Tensor<T>::IsContiguous() const
{
	size_t expected = 1;
	for (size_t i = dimensions.size() - 1; i < dimensions.size(); --i)
	{
		if (strides[i] != expected)
			return false;

		expected *= dimensions[i];
	}
	return true;
}

template<typename T>
inline Tensor<T> Tensor<T>::View(const std::vector<size_t>& Dimensions) const
{
	size_t viewSize = std::accumulate(Dimensions.begin(), Dimensions.end(), size_t(1), std::multiplies<size_t>{});
	assert(viewSize == data->size());

	assert(IsContiguous());

	std::vector<size_t> viewStrides(Dimensions.size());
	viewStrides.back() = 1;
	for (size_t i = Dimensions.size() - 2; i < Dimensions.size(); --i)
		viewStrides[i] = viewStrides[i + 1] * Dimensions[i + 1];

	return Tensor<T>(data, offset, Dimensions, viewStrides);
}

template<typename T>
inline Tensor<T> Tensor<T>::Slice(size_t Dimension, size_t Start, size_t Length) const
{
	assert(Dimension < dimensions.size());
	assert(Start + Length <= dimensions[Dimension]);

	std::vector<size_t> sliceDimensions = dimensions;
	sliceDimensions[Dimension] = Length;

	size_t sliceOffset = offset + Start * strides[Dimension];

	return Tensor<T>(data, sliceOffset, sliceDimensions, strides);
}


