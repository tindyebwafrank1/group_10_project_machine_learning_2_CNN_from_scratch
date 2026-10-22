#include "../include/tensor.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
// construction

Tensor::Tensor() : data(), shape{0, 0, 0, 0} {}

Tensor::Tensor(std::size_t n, std::size_t c, std::size_t h, std::size_t w)
    : data(checkedSize(n, c, h, w), 0.0), shape{n, c, h, w} {}

Tensor Tensor::filled(std::size_t n, std::size_t c, std::size_t h, std::size_t w, double value) {
    Tensor t(n, c, h, w);
    t.fill(value);
    return t;
}

Tensor Tensor::zeros(std::size_t n, std::size_t c, std::size_t h, std::size_t w) {
    return Tensor(n, c, h, w);
}

// shape queries

std::size_t Tensor::size() const noexcept { return data.size(); }

std::size_t Tensor::batch()    const { return shape.at(0); }
std::size_t Tensor::channels() const { return shape.at(1); }
std::size_t Tensor::height()   const { return shape.at(2); }
std::size_t Tensor::width()    const { return shape.at(3); }

std::string Tensor::shapeString() const {
    std::string s = "{";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) s += ",";
        s += std::to_string(shape[i]);
    }
    return s + "}";
}

bool Tensor::isValid() const noexcept {
    if (shape.size() != 4) return false;
    std::size_t total = 1;
    for (std::size_t d : shape) {
        if (d == 0) return false;
        if (total > std::numeric_limits<std::size_t>::max() / d) return false;
        total *= d;
    }
    return data.size() == total;
}

void Tensor::validate(const char* who) const {
    if (shape.size() != 4) {
        throw std::invalid_argument(std::string(who) + ": tensor must be 4-D {N,C,H,W}");
    }
    std::size_t total = 1;
    for (std::size_t d : shape) {
        if (d == 0) throw std::invalid_argument(std::string(who) + ": tensor has a zero dimension");
        if (total > std::numeric_limits<std::size_t>::max() / d) {
            throw std::invalid_argument(std::string(who) + ": tensor too large");
        }
        total *= d;
    }
    if (data.size() != total) {
        throw std::invalid_argument(std::string(who) + ": data size " + std::to_string(data.size()) +
                                    " does not match shape " + shapeString());
    }
}

// indexing

std::size_t Tensor::index(std::size_t n, std::size_t c, std::size_t h, std::size_t w) const noexcept {
    return ((n * shape[1] + c) * shape[2] + h) * shape[3] + w;
}

double& Tensor::operator()(std::size_t n, std::size_t c, std::size_t h, std::size_t w) {
    assert(inRange(n, c, h, w));
    return data[index(n, c, h, w)];
}

const double& Tensor::operator()(std::size_t n, std::size_t c, std::size_t h, std::size_t w) const {
    assert(inRange(n, c, h, w));
    return data[index(n, c, h, w)];
}

double& Tensor::at(std::size_t n, std::size_t c, std::size_t h, std::size_t w) {
    requireInRange(n, c, h, w);
    return data[index(n, c, h, w)];
}

const double& Tensor::at(std::size_t n, std::size_t c, std::size_t h, std::size_t w) const {
    requireInRange(n, c, h, w);
    return data[index(n, c, h, w)];
}

// statistics

void Tensor::fill(double value) {
    std::fill(data.begin(), data.end(), value);
}

double Tensor::sum() const {
    double s = 0.0;
    for (double v : data) s += v;
    return s;
}

double Tensor::mean() const {
    requireNotEmpty("mean");
    return sum() / static_cast<double>(data.size());
}

double Tensor::min() const {
    requireNotEmpty("min");
    return *std::min_element(data.begin(), data.end());
}

double Tensor::max() const {
    requireNotEmpty("max");
    return *std::max_element(data.begin(), data.end());
}

bool Tensor::allFinite() const noexcept {
    for (double v : data) {
        if (!std::isfinite(v)) return false;
    }
    return true;
}

//  arithmetic

Tensor& Tensor::operator+=(const Tensor& rhs) {
    requireSameShape(rhs, "operator+=");
    for (std::size_t i = 0; i < data.size(); ++i) data[i] += rhs.data[i];
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& rhs) {
    requireSameShape(rhs, "operator-=");
    for (std::size_t i = 0; i < data.size(); ++i) data[i] -= rhs.data[i];
    return *this;
}

Tensor& Tensor::operator*=(double scalar) {
    for (double& v : data) v *= scalar;
    return *this;
}

Tensor& Tensor::operator/=(double scalar) {
    if (scalar == 0.0) throw std::invalid_argument("Tensor::operator/=: division by zero");
    for (double& v : data) v /= scalar;
    return *this;
}

Tensor Tensor::add(const Tensor& other) const {
    requireSameShape(other, "add");
    Tensor out = *this;
    for (std::size_t i = 0; i < data.size(); ++i) out.data[i] += other.data[i];
    return out;
}

Tensor Tensor::subtract(const Tensor& other) const {
    requireSameShape(other, "subtract");
    Tensor out = *this;
    for (std::size_t i = 0; i < data.size(); ++i) out.data[i] -= other.data[i];
    return out;
}

Tensor Tensor::multiply(const Tensor& other) const {
    requireSameShape(other, "multiply");
    Tensor out = *this;
    for (std::size_t i = 0; i < data.size(); ++i) out.data[i] *= other.data[i];
    return out;
}

Tensor Tensor::scale(double factor) const {
    Tensor out = *this;
    out *= factor;
    return out;
}

//  reshaping and slicing

Tensor Tensor::reshaped(std::size_t n, std::size_t c, std::size_t h, std::size_t w) const {
    const std::size_t total = checkedSize(n, c, h, w);
    if (total != data.size()) {
        throw std::invalid_argument("Tensor::reshaped: cannot reshape " + shapeString() + " (" +
                                    std::to_string(data.size()) + " elements) to {" + std::to_string(n) + "," +
                                    std::to_string(c) + "," + std::to_string(h) + "," + std::to_string(w) +
                                    "} (" + std::to_string(total) + " elements)");
    }
    Tensor out = *this;
    out.shape = {n, c, h, w};
    return out;
}

Tensor& Tensor::reshape(std::size_t n, std::size_t c, std::size_t h, std::size_t w) {
    *this = reshaped(n, c, h, w);
    return *this;
}

Tensor Tensor::flattened() const {
    validate("Tensor::flattened");
    return reshaped(shape[0], data.size() / shape[0], 1, 1);
}

Tensor Tensor::sample(std::size_t n) const {
    return batchSlice(n, 1);
}

Tensor Tensor::batchSlice(std::size_t first, std::size_t count) const {
    validate("Tensor::batchSlice");
    if (count == 0) throw std::invalid_argument("Tensor::batchSlice: count must be positive");
    if (first >= shape[0] || count > shape[0] - first) {
        throw std::out_of_range("Tensor::batchSlice: images " + std::to_string(first) + " to " +
                                std::to_string(first + count - 1) + " are outside a batch of " +
                                std::to_string(shape[0]));
    }
    const std::size_t per_image = data.size() / shape[0];
    Tensor out(count, shape[1], shape[2], shape[3]);
    std::copy(data.begin() + static_cast<std::ptrdiff_t>(first * per_image),
              data.begin() + static_cast<std::ptrdiff_t>((first + count) * per_image), out.data.begin());
    return out;
}

// comparison

bool Tensor::operator==(const Tensor& other) const {
    return shape == other.shape && data == other.data;
}

bool Tensor::operator!=(const Tensor& other) const {
    return !(*this == other);
}

bool Tensor::approxEqual(const Tensor& other, double eps) const {
    if (shape != other.shape || data.size() != other.data.size()) return false;
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (!(std::fabs(data[i] - other.data[i]) <= eps)) return false;
    }
    return true;
}

//  private

std::size_t Tensor::checkedSize(std::size_t n, std::size_t c, std::size_t h, std::size_t w) {
    if (n == 0 || c == 0 || h == 0 || w == 0) {
        throw std::invalid_argument("Tensor: every dimension must be positive, got {" + std::to_string(n) + "," +
                                    std::to_string(c) + "," + std::to_string(h) + "," + std::to_string(w) + "}");
    }
    const std::size_t dims[4] = {n, c, h, w};
    std::size_t total = 1;
    for (std::size_t d : dims) {
        if (total > std::numeric_limits<std::size_t>::max() / d) {
            throw std::invalid_argument("Tensor: shape is too large (the number of elements overflows)");
        }
        total *= d;
    }
    return total;
}

bool Tensor::inRange(std::size_t n, std::size_t c, std::size_t h, std::size_t w) const noexcept {
    return shape.size() == 4 && n < shape[0] && c < shape[1] && h < shape[2] && w < shape[3];
}

void Tensor::requireInRange(std::size_t n, std::size_t c, std::size_t h, std::size_t w) const {
    if (!inRange(n, c, h, w)) {
        throw std::out_of_range("Tensor::at: index (" + std::to_string(n) + "," + std::to_string(c) + "," +
                                std::to_string(h) + "," + std::to_string(w) + ") is outside shape " +
                                shapeString());
    }
}

void Tensor::requireSameShape(const Tensor& other, const char* who) const {
    if (shape != other.shape || data.size() != other.data.size()) {
        throw std::invalid_argument(std::string("Tensor::") + who + ": shape mismatch " + shapeString() +
                                    " and " + other.shapeString());
    }
}

void Tensor::requireNotEmpty(const char* who) const {
    if (data.empty()) throw std::invalid_argument(std::string("Tensor::") + who + ": tensor has no elements");
}