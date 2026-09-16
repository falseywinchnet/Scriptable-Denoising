#pragma once

// Exact online provider for adaptive dyadic block totals V(B,k).
//
// For block length L, e=N/L, and k=p*e+delta,
//
//   V(B,k) = W_N^(k*lo) F_L^(delta/e)[p].
//
// A twisted FFT inherits alpha=delta/e through even/odd recursion.  One
// butterfly indexed by r produces outputs r and r+M/2 together.  Caching that
// pair makes output requests incremental and order-independent: the final
// number of opened cells is exactly the usual output-pruned FFT recurrence.
//
// No transcendental is needed in a channel.  At recursion length M,
//
//   exp(-2pi i (r+delta/e)/M)
//     = W_N^((r*e+delta)*(L/M)),
//
// an ordinary frame root supplied by one plan-wide table.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fct {
namespace fractional {

struct value {
    double re = 0.0;
    double im = 0.0;
};

inline value add(value a, value b) {
    return value{a.re + b.re, a.im + b.im};
}

inline value sub(value a, value b) {
    return value{a.re - b.re, a.im - b.im};
}

inline value mul(value a, value b) {
    return value{a.re * b.re - a.im * b.im,
                 a.re * b.im + a.im * b.re};
}

struct pair_value {
    value low;
    value high;
};

class dense_cell_arena {
public:
    explicit dense_cell_arena(std::size_t reserve_cells = 0) {
        cells_.reserve(reserve_cells);
        valid_.reserve(reserve_cells);
    }

    std::size_t allocate(std::size_t count) {
        const std::size_t offset = cells_.size();
        cells_.resize(offset + count);
        valid_.resize(offset + count, 0);
        return offset;
    }

    bool valid(std::size_t index) const { return valid_[index] != 0; }
    const pair_value& get(std::size_t index) const { return cells_[index]; }
    void set(std::size_t index, pair_value value) {
        cells_[index] = value;
        valid_[index] = 1;
    }
    std::size_t allocated() const noexcept { return cells_.size(); }

private:
    std::vector<pair_value> cells_;
    std::vector<std::uint8_t> valid_;
};

class incremental_channel {
public:
    incremental_channel(const value* block, int frame_n, int length,
                        int delta, const value* roots, dense_cell_arena* arena)
        : block_(block), n_(frame_n), length_(length), e_(frame_n / length),
          delta_(delta), roots_(roots), arena_(arena) {
        int levels = 0;
        for (int m = length_; m > 1; m >>= 1) ++levels;
        const std::size_t capacity =
            static_cast<std::size_t>(length_ / 2) * levels;
        arena_offset_ = arena_->allocate(capacity);
    }

    std::size_t marginal(int p) const {
        return marginal_rec(0, 0, p);
    }

    value output(int p) { return output_rec(0, 0, p); }

    std::size_t cells() const noexcept { return cell_count_; }

private:
    std::size_t cell_key(int depth, int start, int length, int r) const {
        // Every depth owns length_/2 possible butterfly cells.  ``start`` is
        // the decimated subsequence number and each owns length/2 r values.
        return static_cast<std::size_t>(depth) * (length_ / 2) +
               static_cast<std::size_t>(start) * (length / 2) +
               static_cast<std::size_t>(r);
    }

    std::size_t marginal_rec(int depth, int start, int p) const {
        const int length = length_ >> depth;
        if (length == 1) return 0;
        const int half = length / 2;
        const int r = p % half;
        const std::size_t key = cell_key(depth, start, length, r);
        if (arena_->valid(arena_offset_ + key)) return 0;
        return 1 + marginal_rec(depth + 1, start, r) +
               marginal_rec(depth + 1, start + (1 << depth), r);
    }

    value output_rec(int depth, int start, int p) {
        const int length = length_ >> depth;
        if (length == 1) return block_[start];
        const int half = length / 2;
        const int r = p % half;
        const std::size_t key = cell_key(depth, start, length, r);
        const std::size_t arena_key = arena_offset_ + key;
        if (!arena_->valid(arena_key)) {
            const value even = output_rec(depth + 1, start, r);
            const value odd = output_rec(
                depth + 1, start + (1 << depth), r);
            const std::uint64_t exponent =
                (static_cast<std::uint64_t>(r) * e_ + delta_) *
                static_cast<std::uint64_t>(length_ / length);
            const value z = mul(roots_[exponent % static_cast<unsigned>(n_)],
                                odd);
            arena_->set(arena_key, pair_value{add(even, z), sub(even, z)});
            ++cell_count_;
        }
        const pair_value& pair = arena_->get(arena_key);
        return p < half ? pair.low : pair.high;
    }

    const value* block_;
    int n_;
    int length_;
    int e_;
    int delta_;
    const value* roots_;
    dense_cell_arena* arena_;
    std::size_t arena_offset_ = 0;
    std::size_t cell_count_ = 0;
};

struct channel_key {
    int lo;
    int length;
    int delta;

    bool operator==(const channel_key& other) const noexcept {
        return lo == other.lo && length == other.length &&
               delta == other.delta;
    }
};

struct value_key {
    int lo;
    int length;
    int k;

    bool operator==(const value_key& other) const noexcept {
        return lo == other.lo && length == other.length && k == other.k;
    }
};

struct triple_hash {
    template <typename Key>
    std::size_t operator()(const Key& key) const noexcept {
        std::size_t h = static_cast<std::size_t>(key.lo);
        h ^= static_cast<std::size_t>(key.length) + 0x9e3779b97f4a7c15ULL +
             (h << 6) + (h >> 2);
        const int third = third_of(key);
        h ^= static_cast<std::size_t>(third) + 0x9e3779b97f4a7c15ULL +
             (h << 6) + (h >> 2);
        return h;
    }

private:
    static int third_of(const channel_key& key) noexcept { return key.delta; }
    static int third_of(const value_key& key) noexcept { return key.k; }
};

class provider {
public:
    provider(const value* input, int n, bool cache_values = true)
        : input_(input), n_(n), cache_values_(cache_values),
          cell_arena_(static_cast<std::size_t>(n) * 128) {
        roots_.resize(static_cast<std::size_t>(n_));
        constexpr double tau = 6.283185307179586476925286766559005768;
        for (int j = 0; j < n_; ++j) {
            const double a = -tau * static_cast<double>(j) / n_;
            roots_[static_cast<std::size_t>(j)] = value{std::cos(a), std::sin(a)};
        }
        channels_.reserve(static_cast<std::size_t>(n_) * 4);
        if (cache_values_) values_.reserve(static_cast<std::size_t>(n_) * 32);
    }

    value get(int lo, int length, int k) {
        k %= n_;
        if (k < 0) k += n_;
        const value_key vk{lo, length, k};
        if (cache_values_) {
            auto cached = values_.find(vk);
            if (cached != values_.end()) return cached->second;
        }

        value out;
        if (length == 1) {
            const std::uint64_t exponent =
                static_cast<std::uint64_t>(k) * static_cast<unsigned>(lo);
            out = mul(input_[lo], roots_[exponent % static_cast<unsigned>(n_)]);
            ++leaf_values_;
        } else {
            const int e = n_ / length;
            const int delta = k % e;
            const int p = (k - delta) / e;
            const channel_key ck{lo, length, delta};
            auto channel = channels_.find(ck);
            if (channel == channels_.end()) {
                channel = channels_.try_emplace(
                    ck, input_ + lo, n_, length, delta,
                    roots_.data(), &cell_arena_).first;
            }
            const std::size_t cells_before = channel->second.cells();
            out = channel->second.output(p);
            butterfly_cells_ += channel->second.cells() - cells_before;
            if (lo) {
                const std::uint64_t exponent =
                    static_cast<std::uint64_t>(k) * static_cast<unsigned>(lo);
                out = mul(out, roots_[exponent % static_cast<unsigned>(n_)]);
                ++phase_multiplies_;
            }
            ++direct_values_;
        }
        if (cache_values_) values_.emplace(vk, out);
        ++computed_values_;
        return out;
    }

    std::size_t butterfly_cells() const {
        return butterfly_cells_;
    }

    std::size_t channel_count() const noexcept { return channels_.size(); }
    std::size_t value_count() const noexcept {
        return cache_values_ ? values_.size() : computed_values_;
    }
    std::size_t direct_values() const noexcept { return direct_values_; }
    std::size_t leaf_values() const noexcept { return leaf_values_; }
    std::size_t phase_multiplies() const noexcept { return phase_multiplies_; }

private:
    const value* input_;
    int n_;
    bool cache_values_;
    std::vector<value> roots_;
    dense_cell_arena cell_arena_;
    std::unordered_map<channel_key, incremental_channel, triple_hash> channels_;
    std::unordered_map<value_key, value, triple_hash> values_;
    std::size_t direct_values_ = 0;
    std::size_t leaf_values_ = 0;
    std::size_t phase_multiplies_ = 0;
    std::size_t computed_values_ = 0;
    std::size_t butterfly_cells_ = 0;
};

} // namespace fractional
} // namespace fct
