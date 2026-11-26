#pragma once

#include <type_traits>
#include "hsys/vector.cuh"

namespace hsys {

template <typename AtomT>
requires std::is_arithmetic_v<AtomT>
AtomT vecred_nobr(const Vector<AtomT>& vec);

template <typename AtomT>
requires std::is_arithmetic_v<AtomT>
AtomT vecred_br(const Vector<AtomT>& vec);

template <typename AtomT>
requires std::is_arithmetic_v<AtomT>
void vecred_nobr_device(VectorView<AtomT> vec, AtomT* d_out);

template <typename AtomT>
requires std::is_arithmetic_v<AtomT>
void vecred_br_device(VectorView<AtomT> vec, AtomT* d_out);

} // namespace hsys
