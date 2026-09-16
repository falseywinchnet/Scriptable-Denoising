#pragma once

// Joined intrinsic FCT research kernel.
//
// Frequency bins are SIMD lanes.  Each lane owns a depth-first support stack,
// but bins currently visiting the same dyadic time node are gathered into one
// structure-of-arrays packet.  The packet applies the certified half-edge disk
// mask, and surviving internal lanes request V(left,k) from the incremental
// fractional provider.  No dense prefix bank or heuristic support finder is
// used; activity is a certified incumbent floor.

#include "fct_fractional_provider.hpp"
#include "fct_half_edge_simd.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fct {
namespace intrinsic {

using fractional::value;

struct statistics {
    std::size_t nodes = 0;
    std::size_t mask_packets = 0;
    std::size_t mask_lanes = 0;
    std::size_t max_packet = 0;
    std::size_t rounds = 0;
    std::size_t provider_values = 0;
    std::size_t provider_channels = 0;
    std::size_t provider_cells = 0;
    std::size_t provider_leaf_values = 0;
    std::size_t provider_phase_multiplies = 0;
};

inline double norm2(value z) { return z.re * z.re + z.im * z.im; }

struct task {
    int lo = 0;
    int length = 0;
    value before{};
    value total{};
};

class plan {
public:
    explicit plan(int n) : n_(n), levels_(ilog2_pow2(n)) {
        const std::size_t bins = static_cast<std::size_t>(n_);
        energy_.resize(bins + 1);
        root_.resize(bins);
        best_.resize(bins);
        best_score_.resize(bins);
        best_tau_.resize(bins);
        current_.resize(bins);
        has_current_.resize(bins);
        stack_pos_.resize(bins);
        deferred_.resize(bins * static_cast<std::size_t>(levels_ + 1));
        groups_.resize(2 * bins - 1);
        touched_.reserve(2 * bins - 1);
        lane_br_.resize(bins);
        lane_bi_.resize(bins);
        lane_vr_.resize(bins);
        lane_vi_.resize(bins);
        lane_threshold_.resize(bins);
        lane_mask_.resize(bins);
        moment_input_.resize(bins);
    }

    int size() const noexcept { return n_; }
    const statistics& stats() const noexcept { return stats_; }

    // activity_factor is in units of mean |x|^2.  Zero computes the exact
    // argmax for every bin.  A negative value selects the research floor
    // log(N)+2, below which the ordinary full-support Fourier value is emitted.
    bool forward(const value* input, value* output, std::int64_t* tau,
                 double activity_factor = -1.0, int min_tau = 1,
                 value* moment = nullptr) {
        if (!input || !output || n_ < 2) return false;
        min_tau = std::max(1, std::min(n_, min_tau));
        stats_ = statistics{};

        energy_[0] = 0.0;
        for (int i = 0; i < n_; ++i)
            energy_[static_cast<std::size_t>(i + 1)] =
                energy_[static_cast<std::size_t>(i)] + norm2(input[i]);
        const double mean_energy = energy_.back() / n_;
        const double factor = activity_factor < 0.0
            ? std::log(static_cast<double>(n_)) + 2.0
            : activity_factor;
        const double floor_score = factor * mean_energy;

        fractional::provider provider(input, n_, false);
        for (int k = 0; k < n_; ++k) {
            root_[static_cast<std::size_t>(k)] = provider.get(0, n_, k);
            best_[static_cast<std::size_t>(k)] = root_[static_cast<std::size_t>(k)];
            best_score_[static_cast<std::size_t>(k)] =
                norm2(root_[static_cast<std::size_t>(k)]) / n_;
            best_tau_[static_cast<std::size_t>(k)] = n_;
            current_[static_cast<std::size_t>(k)] =
                task{0, n_, value{}, root_[static_cast<std::size_t>(k)]};
            has_current_[static_cast<std::size_t>(k)] = 1;
            stack_pos_[static_cast<std::size_t>(k)] = 0;
        }

        int remaining = n_;
        while (remaining > 0) {
            ++stats_.rounds;
            touched_.clear();
            for (int k = 0; k < n_; ++k) {
                if (!has_current_[static_cast<std::size_t>(k)]) continue;
                const task& q = current_[static_cast<std::size_t>(k)];
                const std::size_t id = node_id(q.lo, q.length);
                if (groups_[id].empty()) touched_.push_back(id);
                groups_[id].push_back(k);
            }

            for (std::size_t id : touched_) {
                std::vector<int>& bins = groups_[id];
                const task prototype = current_[static_cast<std::size_t>(bins[0])];
                const int lo = prototype.lo;
                const int length = prototype.length;
                const double block_energy =
                    energy_[static_cast<std::size_t>(lo + length)] -
                    energy_[static_cast<std::size_t>(lo)];
                const std::size_t count = bins.size();
                stats_.nodes += count;
                ++stats_.mask_packets;
                stats_.mask_lanes += count;
                stats_.max_packet = std::max(stats_.max_packet, count);

                for (std::size_t lane = 0; lane < count; ++lane) {
                    const int k = bins[lane];
                    const task& q = current_[static_cast<std::size_t>(k)];
                    lane_br_[lane] = q.before.re;
                    lane_bi_[lane] = q.before.im;
                    lane_vr_[lane] = q.total.re;
                    lane_vi_[lane] = q.total.im;
                    lane_threshold_[lane] = std::max(
                        best_score_[static_cast<std::size_t>(k)], floor_score);
                }
                half_edge::prune_same_node(
                    lane_br_.data(), lane_bi_.data(), lane_vr_.data(),
                    lane_vi_.data(), lane_threshold_.data(), count,
                    block_energy, length, lo, lane_mask_.data());

                for (std::size_t lane = 0; lane < count; ++lane) {
                    const int k = bins[lane];
                    const task q = current_[static_cast<std::size_t>(k)];
                    // This node represents prefix endpoints
                    // [lo + 1, lo + length].  A minimum aperture changes only
                    // the feasible domain; discarding a wholly infeasible node
                    // is exact and still uses the same certified disk walk.
                    if (q.lo + q.length < min_tau) {
                        finish_lane(k, remaining);
                        continue;
                    }
                    const double threshold = lane_threshold_[lane];
                    const double energy_r2 = length * block_energy;
                    const double limit2 = threshold * (lo + 1.0);
                    const bool energy_prune = half_edge::prunable_disk_scalar(
                        q.before.re, q.before.im, energy_r2, limit2);
                    if (lane_mask_[lane] || energy_prune) {
                        finish_lane(k, remaining);
                        continue;
                    }

                    if (length == 1) {
                        const value z = fractional::add(q.before, q.total);
                        const double score = norm2(z) / (lo + 1.0);
                        if (score > best_score_[static_cast<std::size_t>(k)]) {
                            best_score_[static_cast<std::size_t>(k)] = score;
                            best_[static_cast<std::size_t>(k)] = z;
                            best_tau_[static_cast<std::size_t>(k)] = lo + 1;
                        }
                        finish_lane(k, remaining);
                        continue;
                    }

                    const int half = length / 2;
                    const value left_total = provider.get(lo, half, k);
                    const value right_total = fractional::sub(q.total, left_total);
                    const value right_before = fractional::add(q.before, left_total);
                    const double left_score = norm2(right_before) / (lo + half);
                    const double right_score =
                        norm2(fractional::add(q.before, q.total)) / (lo + length);
                    const task left{lo, half, q.before, left_total};
                    const task right{lo + half, half, right_before, right_total};
                    if (right_score > left_score) {
                        push_deferred(k, left);
                        current_[static_cast<std::size_t>(k)] = right;
                    } else {
                        push_deferred(k, right);
                        current_[static_cast<std::size_t>(k)] = left;
                    }
                }
                bins.clear();
            }
        }

        for (int k = 0; k < n_; ++k) {
            if (best_score_[static_cast<std::size_t>(k)] > floor_score) {
                output[k] = best_[static_cast<std::size_t>(k)];
                if (tau) tau[k] = best_tau_[static_cast<std::size_t>(k)];
            } else {
                output[k] = root_[static_cast<std::size_t>(k)];
                if (tau) tau[k] = n_;
            }
        }
        if (moment) {
            // The frequency derivative of phase needs
            // M(k,tau)=sum_{t<tau} t*x[t]*exp(-i*omega_k*t).  Evaluate it at
            // the already-selected support, exactly, by decomposing that
            // prefix into O(log N) dyadic blocks served by the same fractional
            // provider abstraction.  Selection is not rerun on t*x.
            for (int t = 0; t < n_; ++t) {
                moment_input_[static_cast<std::size_t>(t)] = value{
                    input[t].re * t, input[t].im * t};
            }
            fractional::provider moment_provider(
                moment_input_.data(), n_, false);
            for (int k = 0; k < n_; ++k) {
                const int selected = best_score_[static_cast<std::size_t>(k)] >
                        floor_score
                    ? best_tau_[static_cast<std::size_t>(k)] : n_;
                value acc{};
                int lo = 0;
                int remaining_tau = selected;
                while (remaining_tau > 0) {
                    int length = 1;
                    while ((length << 1) <= remaining_tau) length <<= 1;
                    acc = fractional::add(
                        acc, moment_provider.get(lo, length, k));
                    lo += length;
                    remaining_tau -= length;
                }
                moment[k] = acc;
            }
        }
        stats_.provider_values = provider.value_count();
        stats_.provider_channels = provider.channel_count();
        stats_.provider_cells = provider.butterfly_cells();
        stats_.provider_leaf_values = provider.leaf_values();
        stats_.provider_phase_multiplies = provider.phase_multiplies();
        return true;
    }

private:
    static int ilog2_pow2(int n) {
        int out = 0;
        while ((1 << out) < n) ++out;
        return out;
    }

    std::size_t node_id(int lo, int length) const {
        const int depth = levels_ - ilog2_pow2(length);
        return (static_cast<std::size_t>(1) << depth) - 1 + lo / length;
    }

    void push_deferred(int k, const task& q) {
        int& pos = stack_pos_[static_cast<std::size_t>(k)];
        deferred_[static_cast<std::size_t>(k) * (levels_ + 1) + pos] = q;
        ++pos;
    }

    void finish_lane(int k, int& remaining) {
        int& pos = stack_pos_[static_cast<std::size_t>(k)];
        if (pos > 0) {
            --pos;
            current_[static_cast<std::size_t>(k)] =
                deferred_[static_cast<std::size_t>(k) * (levels_ + 1) + pos];
        } else {
            has_current_[static_cast<std::size_t>(k)] = 0;
            --remaining;
        }
    }

    int n_;
    int levels_;
    statistics stats_{};
    std::vector<double> energy_;
    std::vector<value> root_;
    std::vector<value> best_;
    std::vector<double> best_score_;
    std::vector<std::int64_t> best_tau_;
    std::vector<task> current_;
    std::vector<std::uint8_t> has_current_;
    std::vector<int> stack_pos_;
    std::vector<task> deferred_;
    std::vector<std::vector<int>> groups_;
    std::vector<std::size_t> touched_;
    std::vector<double> lane_br_, lane_bi_, lane_vr_, lane_vi_, lane_threshold_;
    std::vector<std::uint8_t> lane_mask_;
    std::vector<value> moment_input_;
};

} // namespace intrinsic
} // namespace fct
