#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>

#include "hsys/data.cuh"
#include "hsys/matrix_view.cuh"

namespace hsys {

template <typename AtomT>
requires std::is_arithmetic_v<AtomT>
class Matrix {
private:
    std::shared_ptr<Data<AtomT>> data_;
    MatrixView<AtomT> view_;

public:
    explicit Matrix(std::size_t nrows, std::size_t ncols)
        : data_(std::make_shared<Data<AtomT>>(nrows * ncols)),
          view_(data_->data(), nrows, ncols) {}

    std::size_t size() const noexcept { return view_.size(); }
    std::size_t nrows() const noexcept { return view_.nrows(); }
    std::size_t ncols() const noexcept { return view_.ncols(); }

    Data<AtomT>& data() noexcept { return *data_; }
    const Data<AtomT>& data() const noexcept { return *data_; }

    MatrixView<AtomT>& view() noexcept { return view_; }
    const MatrixView<AtomT>& view() const noexcept { return view_; }
};

template <typename AtomT>
requires std::is_arithmetic_v<AtomT>
Matrix<AtomT> operator*(const Matrix<AtomT>& lhs, const Matrix<AtomT>& rhs);

} // namespace hsys
