#pragma once

#include "hsys/data.cuh"
#include "hsys/vector_view.cuh"

#include <cstddef>
#include <memory>
#include <type_traits>

namespace hsys {

template <typename AtomT>
requires std::is_arithmetic_v<AtomT>
class Vector {
private:
    std::shared_ptr<Data<AtomT>> data_;
    VectorView<AtomT> view_;

public:
    explicit Vector(std::size_t size)
        : data_(std::make_shared<Data<AtomT>>(size)),
          view_(data_->data(), data_->size()) {}

    std::size_t size() const noexcept { return view_.size(); }

    Data<AtomT>& data() noexcept { return *data_; }
    const Data<AtomT>& data() const noexcept { return *data_; }

    VectorView<AtomT>& view() noexcept { return view_; }
    const VectorView<AtomT>& view() const noexcept { return view_; }
};

template <typename AtomT>
Vector<AtomT> operator+(const Vector<AtomT>& lhs, const Vector<AtomT>& rhs);

} // namespace hsys
