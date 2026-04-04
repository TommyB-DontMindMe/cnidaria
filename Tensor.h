#pragma once
#include <vector>
#include <numeric>
#include <memory>
#include <cassert>
#include <functional>
#include <initializer_list>

template <typename T>
class Tensor
{
public:
	std::vector<size_t> dimensions;
	std::vector<size_t> strides;
	std::shared_ptr<std::vector<T>> data;
	size_t offset = 0;

	Tensor(std::shared_ptr<std::vector<T>> Data, size_t Offset, const std::vector<size_t>& Dimensions, const std::vector<size_t>& Strides) : data(Data), offset(Offset), dimensions(Dimensions), strides(Strides) {}

	Tensor(const std::vector<size_t>& Dimensions);

	size_t IndexFromCoordinate(const std::vector<size_t>& Coordinates) const;

	T operator[](std::initializer_list<size_t> Coordinates) const { return (*data)[IndexFromCoordinate(std::vector<size_t>(Coordinates))]; }
	T& operator[](std::initializer_list<size_t> Coordinates) { return (*data)[IndexFromCoordinate(std::vector<size_t>(Coordinates))]; }

	const std::vector<std::size_t>& Shape() const { return dimensions; }

	// Create a new Tensor sharing the same data, can be reshaped
	Tensor<T> View(const std::vector<size_t>& Dimensions) const;

	// Dimension: Index of the dimension to query
	Tensor<T> Slice(size_t Dimension, size_t Start, size_t Length) const;

	// Create a new Tensor sharing the same data, but rearrange strides
	// Permutations: Stride indices in the desired order
	Tensor<T> Transpose(const std::vector<size_t>& Permutations) const;

	bool IsContiguous() const;

	size_t LogicalSize() const { return std::accumulate(dimensions.begin(), dimensions.end(), size_t(1), std::multiplies<size_t>{}); }

	static std::vector<size_t> BroadcastShape(const std::vector<size_t>& A, const std::vector<size_t>& B);
	Tensor<T> BroadcastTo(const std::vector<size_t>& TargetShape) const;

	void ApplyBroadcasted(const Tensor<T>& Other, std::function<void(T&, const T&)> Operation);

	// Iterator
	template<bool IsConst>
	class IteratorBase
	{
		using TensorPtr = std::conditional_t<IsConst, const Tensor<T>*, Tensor<T>*>;

		TensorPtr tensor;
		size_t physicalOffset;
		std::vector<size_t> coordinates;
		std::vector<ptrdiff_t> step;
		std::vector<ptrdiff_t> reset;
		size_t remaining;

	public:
		using iterator_category = std::forward_iterator_tag;
		using value_type = T;
		using difference_type = ptrdiff_t;
		using pointer = T*;
		using reference = std::conditional_t<IsConst, const T&, T&>;

		IteratorBase(TensorPtr t) : tensor(t), physicalOffset(t->offset), coordinates(t->dimensions.size(), 0), step(t->dimensions.size()), reset(t->dimensions.size()), remaining(t->LogicalSize()) // Begin
		{
			for (size_t d = 0; d < t->dimensions.size(); ++d)
			{
				step[d] = (ptrdiff_t)t->strides[d];
				reset[d] = -((ptrdiff_t)t->dimensions[d] * (ptrdiff_t)t->strides[d]);
			}
		} 
		IteratorBase(TensorPtr t, bool) : tensor(t), physicalOffset(0), coordinates(t->dimensions.size(), 0), step(t->dimensions.size()), reset(t->dimensions.size()), remaining(0) {} // End
		
		reference operator*() const { return (*tensor->data)[physicalOffset]; }

		IteratorBase& operator++()
		{
			--remaining;
			for (ptrdiff_t d = (ptrdiff_t)coordinates.size() - 1; d >= 0; --d)
			{
				++coordinates[d];
				physicalOffset += step[d];
				if (coordinates[d] < tensor->dimensions[d])
					break;
				coordinates[d] = 0;
				physicalOffset += reset[d];
			}
			return *this;
		}

		IteratorBase operator++(int)
		{
			IteratorBase old = *this;
			++(*this);
			return old;
		}

		bool operator==(const IteratorBase& other) const { return remaining == other.remaining; }
		bool operator!=(const IteratorBase& other) const { return remaining != other.remaining; }
	};

	using Iterator = IteratorBase<false>;
	using ConstIterator = IteratorBase<true>;

	Iterator begin() { return Iterator(this); }
	Iterator end() { return Iterator(this, true); }
	ConstIterator begin() const { return ConstIterator(this); }
	ConstIterator end() const { return ConstIterator(this, true); }

	// Math operations
	void Add(T Addend);
	void Add(const Tensor<T>& Addend);
	Tensor<T>& operator+=(T Addend) { Add(Addend); return *this; }
	Tensor<T>& operator+=(const Tensor<T>& Addend) { Add(Addend); return *this; }
	friend Tensor<T> operator+(Tensor<T> A, const Tensor<T>& B) { A += B; return A; }
	friend Tensor<T> operator+(Tensor<T> A, T B) { A += B; return A; }

	void Subtract(T Subtrahend);
	void Subtract(const Tensor<T>& Subtrahend);
	Tensor<T>& operator-=(T Subtrahend) { Subtract(Subtrahend); return *this; }
	Tensor<T>& operator-=(const Tensor<T>& Subtrahend) { Subtract(Subtrahend); return *this; }
	friend Tensor<T> operator-(Tensor<T> A, const Tensor<T>& B) { A -= B; return A; }
	friend Tensor<T> operator-(Tensor<T> A, T B) { A -= B; return A; }

	void Multiply(T Factor);
	void Multiply(const Tensor<T>& Factor);
	Tensor<T>& operator*=(T Factor) { Multiply(Factor); return *this; }
	Tensor<T>& operator*=(const Tensor<T>& Factor) { Multiply(Factor); return *this; }
	friend Tensor<T> operator*(Tensor<T> A, const Tensor<T>& B) { A *= B; return A; }
	friend Tensor<T> operator*(Tensor<T> A, T B) { A *= B; return A; }

	void Divide(T Divisor);
	void Divide(const Tensor<T>& Divisor);
	Tensor<T>& operator/=(T Divisor) { Divide(Divisor); return *this; }
	Tensor<T>& operator/=(const Tensor<T>& Divisor) { Divide(Divisor); return *this; }
	friend Tensor<T> operator/(Tensor<T> A, const Tensor<T>& B) { A /= B; return A; }
	friend Tensor<T> operator/(Tensor<T> A, T B) { A /= B; return A; }
};

template<typename T>
inline Tensor<T>::Tensor(const std::vector<size_t>& Dimensions) : dimensions(Dimensions), offset(0)
{
	size_t dataSize = LogicalSize();
	data = std::make_shared<std::vector<T>>(dataSize);

	strides.resize(dimensions.size());
	strides.back() = 1;	
	// Iterate from end to beginning, size_t is unsigned so it will go from 0 to the highest possible value
	for (size_t i = dimensions.size() - 2; i < dimensions.size(); --i)
		strides[i] = strides[i + 1] * dimensions[i + 1];
}

template<typename T>
inline size_t Tensor<T>::IndexFromCoordinate(const std::vector<size_t>& Coordinates) const
{
	assert(Coordinates.size() == dimensions.size());
	for (size_t i = 0; i < Coordinates.size(); ++i)
		assert(Coordinates[i] < dimensions[i]);
	
	return offset + std::inner_product(Coordinates.begin(), Coordinates.end(), strides.begin(), size_t(0));
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
inline void Tensor<T>::ApplyBroadcasted(const Tensor<T>& Other, std::function<void(T&, const T&)> Operation)
{
	std::vector<size_t> broadcastShape = BroadcastShape(dimensions, Other.dimensions);
	assert(broadcastShape == dimensions);
	Tensor<T> tensorA = BroadcastTo(broadcastShape);
	Tensor<T> tensorB = Other.BroadcastTo(broadcastShape);

	Tensor<T>::Iterator iteratorA = tensorA.begin();
	Tensor<T>::ConstIterator iteratorB = tensorB.begin();
	for (; iteratorA != tensorA.end(); ++iteratorA, ++iteratorB)
		Operation(*iteratorA, *iteratorB);
}

// Addition
template<typename T>
inline void Tensor<T>::Add(T Addend)
{
	for (T& value : *this)
		value += Addend;
}

template<typename T>
inline void Tensor<T>::Add(const Tensor<T>& Addend)
{
	ApplyBroadcasted(Addend, [](T& a, const T& b) { a += b; });
}

// Subtraction
template<typename T>
inline void Tensor<T>::Subtract(T Subtrahend)
{
	for (T& value : *this)
		value -= Subtrahend;
}

template<typename T>
inline void Tensor<T>::Subtract(const Tensor<T>&Subtrahend)
{
	ApplyBroadcasted(Subtrahend, [](T& a, const T& b) { a -= b; });
}

// Multiplication
template<typename T>
inline void Tensor<T>::Multiply(T Factor)
{
	for (T& value : *this)
		value *= Factor;
}

template<typename T>
inline void Tensor<T>::Multiply(const Tensor<T>& Factor)
{
	ApplyBroadcasted(Factor, [](T& a, const T& b) { a *= b; });
}

// Division
template<typename T>
inline void Tensor<T>::Divide(T Divisor)
{
	for (T& value : *this)
		value /= Divisor;
}

template<typename T>
inline void Tensor<T>::Divide(const Tensor<T>& Divisor)
{
	ApplyBroadcasted(Divisor, [](T& a, const T& b) { a /= b; });
}



template<typename T>
inline Tensor<T> Tensor<T>::View(const std::vector<size_t>& Dimensions) const
{
	size_t viewSize = std::accumulate(Dimensions.begin(), Dimensions.end(), size_t(1), std::multiplies<size_t>{});
	assert(viewSize == LogicalSize());

	assert(IsContiguous());

	std::vector<size_t> viewStrides(Dimensions.size());
	viewStrides.back() = 1;
	// Iterate from end to beginning, size_t is unsigned so it will go from 0 to the highest possible value
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

template<typename T>
inline Tensor<T> Tensor<T>::Transpose(const std::vector<size_t>& Permutations) const
{
	assert(Permutations.size() == dimensions.size());

	std::vector<size_t> newDimensions(dimensions.size());
	std::vector<size_t> newStrides(strides.size());

	for (size_t i = 0; i < Permutations.size(); ++i)
	{
		newDimensions[i] = dimensions[Permutations[i]];
		newStrides[i] = strides[Permutations[i]];
	}
	return Tensor<T>(data, offset, newDimensions, newStrides);
}

template<typename T>
inline std::vector<size_t> Tensor<T>::BroadcastShape(const std::vector<size_t>& A, const std::vector<size_t>& B)
{
	size_t rank = std::max(A.size(), B.size());
	std::vector<size_t> result(rank);

	for (size_t i = 0; i < rank; ++i)
	{
		size_t dimensionA = (i < rank - A.size()) ? 1 : A[i - (rank - A.size())];
		size_t dimensionB = (i < rank - B.size()) ? 1 : B[i - (rank - B.size())];

		assert(dimensionA == dimensionB || dimensionA == 1 || dimensionB == 1);
		result[i] = std::max(dimensionA, dimensionB);
	}
	return result;
}

template<typename T>
inline Tensor<T> Tensor<T>::BroadcastTo(const std::vector<size_t>& TargetShape) const
{
	assert(TargetShape.size() >= dimensions.size());

	size_t rank = TargetShape.size();
	std::vector<size_t> newStrides(rank, 0);

	for (size_t i = 0; i < rank; ++i)
	{
		size_t alignedSourceIndex = i - (rank - dimensions.size());
		if (i < rank - dimensions.size())
		{
			newStrides[i] = 0;
		}
		else
		{
			size_t sourceDimension = dimensions[alignedSourceIndex];
			assert(sourceDimension == TargetShape[i] || sourceDimension == 1);
			newStrides[i] = (sourceDimension == 1) ? 0 : strides[alignedSourceIndex];
		}
	}
	return Tensor<T>(data, offset, TargetShape, newStrides);
}

