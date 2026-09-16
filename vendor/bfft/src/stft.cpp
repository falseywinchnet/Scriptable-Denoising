#include <bfft/stft.h>
#include <bfft/bodft.h>
#include <bfft/fct.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <vector>

struct bfft_stft_plan {
    size_t n = 0;
    size_t n_fft = 0;
    size_t hop = 0;
    size_t bins = 0;
    size_t segments = 0;
    bfft_stft_transform transform = BFFT_STFT_RFFT;

    bfft_plan* rfft = nullptr;
    bfft_workspace* rfft_workspace = nullptr;
    bodft_plan* odft = nullptr;
    fct_plan* fct = nullptr;

    std::vector<double> window;
    std::vector<double> analysis;
    std::vector<double> synthesis;
    std::vector<double> buffer;
    std::vector<double> xp;
    std::vector<double> segment;
    std::vector<double> frame;
    std::vector<double> processed;
    std::vector<bfft_complex> tmp_bins;

    ~bfft_stft_plan() {
        bfft_workspace_destroy(rfft_workspace);
        bfft_plan_destroy(rfft);
        bodft_plan_destroy(odft);
        fct_plan_destroy(fct);
    }
};

namespace {

bool is_power_of_two(size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

bool add_overflows_size(size_t a, size_t b) {
    return a > std::numeric_limits<size_t>::max() - b;
}

bool mul_overflows_size(size_t a, size_t b) {
    if (a == 0 || b == 0) {
        return false;
    }
    return a > std::numeric_limits<size_t>::max() / b;
}

int reflect_index(long long i, size_t n) {
    if (n <= 1) return 0;
    const long long period = static_cast<long long>(2 * n - 2);
    long long r = i % period;
    if (r < 0) r += period;
    if (r < static_cast<long long>(n)) return static_cast<int>(r);
    return static_cast<int>(period - r);
}

void ifftshift_copy(const std::vector<double>& input, std::vector<double>& output) {
    const size_t n = input.size();
    const size_t shift = n / 2;
    for (size_t i = 0; i < n; ++i) output[i] = input[(i + shift) % n];
}

bool synthesis_window(const std::vector<double>& window, size_t hop, std::vector<double>& synthesis) {
    const size_t n_fft = window.size();
    std::vector<double> den(n_fft, 0.0);
    const size_t kmax = n_fft / hop;
    for (long long k = -static_cast<long long>(kmax); k <= static_cast<long long>(kmax); ++k) {
        const long long shift = k * static_cast<long long>(hop);
        for (size_t i = 0; i < n_fft; ++i) {
            const long long j = static_cast<long long>(i) - shift;
            if (j >= 0 && j < static_cast<long long>(n_fft)) {
                const double v = window[static_cast<size_t>(j)];
                den[i] += v * v;
            }
        }
    }
    synthesis.resize(n_fft);
    for (size_t i = 0; i < n_fft; ++i) {
        if (!(den[i] > 0.0)) return false;
        synthesis[i] = window[i] / den[i];
    }
    return true;
}

void fill_reflect_pad(const double* input, bfft_stft_plan* plan) {
    const size_t half = plan->n_fft / 2;
    for (size_t i = 0; i < plan->xp.size(); ++i) {
        long long j = static_cast<long long>(i) - static_cast<long long>(half);
        plan->xp[i] = input[reflect_index(j, plan->n)];
    }
}

void overlap_add_tail(bfft_stft_plan* plan, double* output, size_t pos) {
    const size_t hop = plan->hop;
    const size_t blen = plan->buffer.size();
    const double* proc = plan->processed.data();
    double* buffer = plan->buffer.data();

    if (blen == 0) {
        for (size_t a = 0; a < hop; ++a) output[pos + a] = proc[a];
        return;
    }
    if (hop <= blen) {
        const size_t remain = blen - hop;
        for (size_t a = 0; a < hop; ++a) output[pos + a] = proc[a] + buffer[a];
        for (size_t a = 0; a < remain; ++a) buffer[a] = buffer[a + hop] + proc[hop + a];
        for (size_t a = remain; a < blen; ++a) buffer[a] = proc[hop + a];
        return;
    }
    for (size_t a = 0; a < blen; ++a) output[pos + a] = proc[a] + buffer[a];
    for (size_t a = blen; a < hop; ++a) output[pos + a] = proc[a];
    for (size_t a = 0; a < blen; ++a) buffer[a] = proc[hop + a];
}

} // namespace

bfft_status bfft_stft_hann_window(size_t n_fft, double* output) {
    if (!output || n_fft == 0) return BFFT_ERROR_INVALID_ARGUMENT;
    if (n_fft == 1) {
        output[0] = 1.0;
        return BFFT_OK;
    }
    const double denom = static_cast<double>(n_fft - 1);
    for (size_t i = 0; i < n_fft; ++i) {
        output[i] = 0.5 - 0.5 * std::cos((2.0 * 3.141592653589793238462643383279502884 * static_cast<double>(i)) / denom);
    }
    return BFFT_OK;
}

bfft_status bfft_stft_plan_create(size_t n, size_t n_fft, size_t hop_length, const double* window,
                                  bfft_stft_transform transform, bfft_stft_plan** out_plan) {
    if (!out_plan) return BFFT_ERROR_INVALID_ARGUMENT;
    *out_plan = nullptr;
    if (n == 0 || n_fft < 4 || !is_power_of_two(n_fft) || (n_fft % 2) != 0) return BFFT_ERROR_INVALID_ARGUMENT;
    if (hop_length == 0 || hop_length > n_fft || (n_fft % hop_length) != 0) return BFFT_ERROR_INVALID_ARGUMENT;
    if ((n % hop_length) != 0) return BFFT_ERROR_INVALID_ARGUMENT;
    if (transform != BFFT_STFT_RFFT && transform != BFFT_STFT_ODFT &&
        transform != BFFT_STFT_FCT) return BFFT_ERROR_INVALID_ARGUMENT;
    if (transform == BFFT_STFT_FCT && n_fft < 16) return BFFT_ERROR_INVALID_ARGUMENT;
    if (n > static_cast<size_t>(std::numeric_limits<long long>::max())) return BFFT_ERROR_INVALID_ARGUMENT;
    if (n_fft > (static_cast<size_t>(std::numeric_limits<long long>::max()) + 2) / 2) return BFFT_ERROR_INVALID_ARGUMENT;
    if (n_fft > std::numeric_limits<int>::max()) return BFFT_ERROR_INVALID_ARGUMENT;
    if (add_overflows_size(n, n_fft - 1)) return BFFT_ERROR_INVALID_ARGUMENT;
    if (mul_overflows_size(n / hop_length, n_fft / 2 + 1)) return BFFT_ERROR_INVALID_ARGUMENT;

    try {
        std::unique_ptr<bfft_stft_plan> p(new (std::nothrow) bfft_stft_plan());
        if (!p) return BFFT_ERROR_ALLOCATION;
        p->n = n;
        p->n_fft = n_fft;
        p->hop = hop_length;
        p->segments = n / hop_length;
        p->transform = transform;
        p->window.resize(n_fft);
        if (window) {
            for (size_t i = 0; i < n_fft; ++i) {
                if (!std::isfinite(window[i])) return BFFT_ERROR_INVALID_ARGUMENT;
                p->window[i] = window[i];
            }
        } else {
            bfft_status st = bfft_stft_hann_window(n_fft, p->window.data());
            if (st != BFFT_OK) return st;
        }
        p->analysis.resize(n_fft);
        if (transform == BFFT_STFT_FCT) {
            // Leading-edge frames stay in natural time order: no fftshift
            // centering, so the analysis window is applied unshifted.
            for (size_t i = 0; i < n_fft; ++i) p->analysis[i] = p->window[i];
        } else {
            ifftshift_copy(p->window, p->analysis);
        }
        if (!synthesis_window(p->window, hop_length, p->synthesis)) return BFFT_ERROR_INVALID_ARGUMENT;

        if (transform == BFFT_STFT_RFFT) {
            bfft_status st = bfft_plan_create(n_fft, &p->rfft);
            if (st != BFFT_OK) return st;
            st = bfft_workspace_create(p->rfft, &p->rfft_workspace);
            if (st != BFFT_OK) return st;
            p->bins = bfft_plan_bins(p->rfft);
        } else if (transform == BFFT_STFT_ODFT) {
            bfft_status st = bodft_plan_create(n_fft, &p->odft);
            if (st != BFFT_OK) return st;
            p->bins = bodft_plan_bins(p->odft);
        } else {
            bfft_status st = fct_plan_create(n_fft, &p->fct);
            if (st != BFFT_OK) return st;
            p->bins = fct_plan_bins(p->fct);
        }
        p->buffer.assign(n_fft - hop_length, 0.0);
        p->xp.resize(n + n_fft - 1);
        p->segment.resize(n_fft);
        p->frame.resize(n_fft);
        p->processed.resize(n_fft);
        p->tmp_bins.resize(p->bins);
        *out_plan = p.release();
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
    return BFFT_OK;
}

void bfft_stft_plan_destroy(bfft_stft_plan* plan) { delete plan; }
size_t bfft_stft_plan_n(const bfft_stft_plan* plan) { return plan ? plan->n : 0; }
size_t bfft_stft_plan_n_fft(const bfft_stft_plan* plan) { return plan ? plan->n_fft : 0; }
size_t bfft_stft_plan_hop_length(const bfft_stft_plan* plan) { return plan ? plan->hop : 0; }
size_t bfft_stft_plan_bins(const bfft_stft_plan* plan) { return plan ? plan->bins : 0; }
size_t bfft_stft_plan_segments(const bfft_stft_plan* plan) { return plan ? plan->segments : 0; }
size_t bfft_stft_plan_buffer_length(const bfft_stft_plan* plan) { return plan ? plan->buffer.size() : 0; }
bfft_stft_transform bfft_stft_plan_transform(const bfft_stft_plan* plan) { return plan ? plan->transform : BFFT_STFT_RFFT; }

bfft_status bfft_stft_reset_buffer(bfft_stft_plan* plan) {
    if (!plan) return BFFT_ERROR_INVALID_ARGUMENT;
    std::fill(plan->buffer.begin(), plan->buffer.end(), 0.0);
    return BFFT_OK;
}

bfft_status bfft_stft_forward(bfft_stft_plan* plan, const double* input, bfft_complex* output) {
    if (!plan || !input || !output) return BFFT_ERROR_INVALID_ARGUMENT;
    const size_t half = plan->n_fft / 2;
    fill_reflect_pad(input, plan);
    for (size_t col = 0; col < plan->segments; ++col) {
        const size_t base = col * plan->hop;
        if (plan->transform == BFFT_STFT_FCT) {
            // natural frame order: the leading edge is a time-domain notion
            for (size_t a = 0; a < plan->n_fft; ++a) {
                plan->segment[a] = plan->xp[base + a] * plan->analysis[a];
            }
        } else {
            for (size_t a = 0; a < half; ++a) plan->segment[a] = plan->xp[base + a + half] * plan->analysis[a];
            for (size_t a = 0; a < half; ++a) {
                const size_t j = a + half;
                double v = plan->xp[base + a] * plan->analysis[j];
                if (plan->transform == BFFT_STFT_ODFT) v = -v;
                plan->segment[j] = v;
            }
        }
        bfft_status st;
        if (plan->transform == BFFT_STFT_RFFT) {
            st = bfft_forward_workspace(plan->rfft, plan->rfft_workspace,
                                        plan->segment.data(),
                                        plan->tmp_bins.data());
        } else if (plan->transform == BFFT_STFT_ODFT) {
            st = bodft_forward(plan->odft, plan->segment.data(), plan->tmp_bins.data());
        } else {
            st = fct_forward(plan->fct, plan->segment.data(), plan->tmp_bins.data(), nullptr);
        }
        if (st != BFFT_OK) return st;
        for (size_t bin = 0; bin < plan->bins; ++bin) output[bin * plan->segments + col] = plan->tmp_bins[bin];
    }
    return BFFT_OK;
}

bfft_status bfft_stft_inverse(bfft_stft_plan* plan, const bfft_complex* input, double* output) {
    if (!plan || !input || !output) return BFFT_ERROR_INVALID_ARGUMENT;
    /* The FCT is forward-only: the per-bin slice selection is nonlinear and
       the underlying truncation family is exponentially ill-conditioned. */
    if (plan->transform == BFFT_STFT_FCT) return BFFT_ERROR_INVALID_ARGUMENT;
    const size_t half = plan->n_fft / 2;
    size_t pos = 0;
    for (size_t col = 0; col < plan->segments; ++col) {
        for (size_t bin = 0; bin < plan->bins; ++bin) plan->tmp_bins[bin] = input[bin * plan->segments + col];
        bfft_status st;
        if (plan->transform == BFFT_STFT_RFFT)
            st = bfft_inverse_workspace(plan->rfft, plan->rfft_workspace,
                                        plan->tmp_bins.data(),
                                        plan->frame.data());
        else st = bodft_inverse(plan->odft, plan->tmp_bins.data(), plan->frame.data());
        if (st != BFFT_OK) return st;
        for (size_t a = 0; a < half; ++a) {
            double v = plan->frame[a + half];
            if (plan->transform == BFFT_STFT_ODFT) v = -v;
            plan->processed[a] = v * plan->synthesis[a];
        }
        for (size_t a = 0; a < half; ++a) {
            const size_t j = a + half;
            plan->processed[j] = plan->frame[a] * plan->synthesis[j];
        }
        overlap_add_tail(plan, output, pos);
        pos += plan->hop;
    }
    return BFFT_OK;
}
