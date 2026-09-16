#ifndef BFFT_BFFT_HPP
#define BFFT_BFFT_HPP

#include <bfft/bfft.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace bfft {

/* C++ aliases for the C ABI types. */
using complex = bfft_complex;
using complex_f32 = bfft_complex_f32;
using layout = bfft_layout;
using status = bfft_status;

/* Library version string, for example "0.1.0". */
inline std::string version_string() {
    return bfft_version_string();
}

/* SIMD backend selected by the build target. */
inline std::string backend_name() {
    return bfft_backend_name();
}

/* Exception thrown when a wrapped C API call returns an error status. */
class error : public std::runtime_error {
public:
    explicit error(status code)
        : std::runtime_error(bfft_status_string(code)), code_(code) {}

    status code() const noexcept {
        return code_;
    }

private:
    status code_;
};

class workspace {
public:
    workspace() noexcept = default;

    explicit workspace(const bfft_plan* plan_ptr) {
        bfft_workspace* raw = nullptr;
        status result = bfft_workspace_create(plan_ptr, &raw);
        if (result != BFFT_OK) {
            throw error(result);
        }
        impl_.reset(raw);
    }

private:
    friend class plan;

    struct deleter {
        void operator()(bfft_workspace* workspace_ptr) const noexcept {
            bfft_workspace_destroy(workspace_ptr);
        }
    };

    bfft_workspace* get() const noexcept {
        return impl_.get();
    }

    std::unique_ptr<bfft_workspace, deleter> impl_;
};

/* Reusable real FFT plan. Construction validates the transform size and owns
   the underlying C plan with RAII. */
class plan {
public:
    /* Create a plan for a power-of-two real FFT transform size N >= 4. */
    explicit plan(std::size_t n) {
        bfft_plan* raw = nullptr;
        status result = bfft_plan_create(n, &raw);
        if (result != BFFT_OK) {
            throw error(result);
        }
        impl_.reset(raw);
    }

    /* Transform size N. */
    std::size_t size() const noexcept {
        return bfft_plan_size(impl_.get());
    }

    /* Number of complex r2c bins, N / 2 + 1. */
    std::size_t bins() const noexcept {
        return bfft_plan_bins(impl_.get());
    }

    /* Number of doubles needed by forward work buffers. */
    std::size_t work_size() const noexcept {
        return bfft_plan_work_size(impl_.get());
    }

    /* Number of floats needed by single-precision forward work buffers. */
    std::size_t work_size_f32() const noexcept {
        return bfft_plan_work_size_f32(impl_.get());
    }

    /* Number of complex values to reserve for standard forward scratch. */
    std::size_t native_scratch_size() const noexcept {
        return bfft_plan_native_scratch_size(impl_.get());
    }

    /* Create aligned scratch storage matched to this plan. */
    workspace create_workspace() const {
        return workspace(impl_.get());
    }

    /* Standard-output packing policy chosen for this plan. */
    std::string standard_policy() const {
        return bfft_plan_standard_policy(impl_.get());
    }

    /* Standard FFT-order forward transform. Buffers follow the C API sizes. */
    void forward(const double* input,
                 complex* output,
                 double* work,
                 complex* native_scratch) const {
        check(bfft_forward(impl_.get(), input, output, work, native_scratch));
    }

    /* Standard forward transform using reusable aligned workspace scratch. */
    void forward(const double* input, complex* output,
                 workspace& scratch) const {
        check(bfft_forward_workspace(
            impl_.get(), scratch.get(), input, output));
    }

    /* Convenience standard forward transform that allocates work buffers. */
    std::vector<complex> forward(const std::vector<double>& input) const {
        if (input.size() != size()) {
            throw error(BFFT_ERROR_INVALID_ARGUMENT);
        }
        std::vector<complex> output(bins());
        std::vector<double> work(work_size());
        std::vector<complex> scratch(native_scratch_size());
        forward(input.data(), output.data(), work.data(), scratch.data());
        return output;
    }

    /* Native-order forward transform. */
    void forward_native(const double* input, complex* output, double* work) const {
        check(bfft_forward_native(impl_.get(), input, output, work));
    }

    /* Native-order forward transform using aligned workspace scratch. */
    void forward_native(const double* input, complex* output, workspace& scratch) const {
        check(bfft_forward_native_workspace(impl_.get(), scratch.get(), input, output));
    }

    /* Single-precision standard FFT-order forward transform. */
    void forward_f32(const float* input,
                     complex_f32* output,
                     float* work,
                     complex_f32* native_scratch = nullptr) const {
        check(bfft_forward_f32(impl_.get(), input, output, work, native_scratch));
    }

    /* Convenience single-precision standard forward transform. */
    std::vector<complex_f32> forward_f32(const std::vector<float>& input) const {
        if (input.size() != size()) {
            throw error(BFFT_ERROR_INVALID_ARGUMENT);
        }
        std::vector<complex_f32> output(bins());
        std::vector<float> work(work_size_f32());
        forward_f32(input.data(), output.data(), work.data());
        return output;
    }

    /* Single-precision native-order forward transform. */
    void forward_native_f32(const float* input, complex_f32* output, float* work) const {
        check(bfft_forward_native_f32(impl_.get(), input, output, work));
    }

    /* Standard FFT-order magnitude-only forward transform. */
    void forward_magnitude(const double* input, double* magnitudes, double* work) const {
        check(bfft_forward_magnitude(impl_.get(), input, magnitudes, work));
    }

    /* Convenience magnitude-only forward transform that allocates work. */
    std::vector<double> forward_magnitude(const std::vector<double>& input) const {
        if (input.size() != size()) {
            throw error(BFFT_ERROR_INVALID_ARGUMENT);
        }
        std::vector<double> magnitudes(bins());
        std::vector<double> work(work_size());
        forward_magnitude(input.data(), magnitudes.data(), work.data());
        return magnitudes;
    }

    /* Standard FFT-order real-to-polar forward transform, with bins k = 0..N/2. */
    void forward_mag_phase(const double* input, complex* output, double* work) const {
        check(bfft_forward_mag_phase(impl_.get(), input, output, work));
    }

    /* Convenience FFT-order real-to-polar forward transform that allocates work. */
    std::vector<complex> forward_mag_phase(const std::vector<double>& input) const {
        if (input.size() != size()) {
            throw error(BFFT_ERROR_INVALID_ARGUMENT);
        }
        std::vector<complex> output(bins());
        std::vector<double> work(work_size());
        forward_mag_phase(input.data(), output.data(), work.data());
        return output;
    }

    /* Single-precision standard FFT-order magnitude-only forward transform. */
    void forward_magnitude_f32(const float* input, float* magnitudes, float* work) const {
        check(bfft_forward_magnitude_f32(impl_.get(), input, magnitudes, work));
    }

    /* Convenience single-precision magnitude-only forward transform. */
    std::vector<float> forward_magnitude_f32(const std::vector<float>& input) const {
        if (input.size() != size()) {
            throw error(BFFT_ERROR_INVALID_ARGUMENT);
        }
        std::vector<float> magnitudes(bins());
        std::vector<float> work(work_size_f32());
        forward_magnitude_f32(input.data(), magnitudes.data(), work.data());
        return magnitudes;
    }

    /* Single-precision standard FFT-order real-to-polar forward transform, with bins k = 0..N/2. */
    void forward_mag_phase_f32(const float* input, complex_f32* output, float* work) const {
        check(bfft_forward_mag_phase_f32(impl_.get(), input, output, work));
    }

    /* Convenience single-precision FFT-order real-to-polar forward transform. */
    std::vector<complex_f32> forward_mag_phase_f32(const std::vector<float>& input) const {
        if (input.size() != size()) {
            throw error(BFFT_ERROR_INVALID_ARGUMENT);
        }
        std::vector<complex_f32> output(bins());
        std::vector<float> work(work_size_f32());
        forward_mag_phase_f32(input.data(), output.data(), work.data());
        return output;
    }

    /* Standard FFT-order inverse transform. */
    void inverse(const complex* input, double* output) const {
        check(bfft_inverse(impl_.get(), input, output));
    }

    /* Standard inverse transform using reusable aligned workspace scratch. */
    void inverse(const complex* input, double* output,
                 workspace& scratch) const {
        check(bfft_inverse_workspace(
            impl_.get(), scratch.get(), input, output));
    }

    /* Single-precision standard FFT-order inverse transform. */
    void inverse_f32(const complex_f32* input, float* output) const {
        check(bfft_inverse_f32(impl_.get(), input, output));
    }

    /* Standard FFT-order polar-to-real inverse transform. */
    void inverse_mag_phase(const complex* input, double* output) const {
        check(bfft_inverse_mag_phase(impl_.get(), input, output));
    }

    /* Single-precision standard FFT-order polar-to-real inverse transform. */
    void inverse_mag_phase_f32(const complex_f32* input, float* output) const {
        check(bfft_inverse_mag_phase_f32(impl_.get(), input, output));
    }

    /* Convenience polar-to-real inverse transform. */
    std::vector<double> inverse_mag_phase(const std::vector<complex>& input) const {
        if (input.size() != bins()) {
            throw error(BFFT_ERROR_INVALID_ARGUMENT);
        }
        std::vector<double> output(size());
        inverse_mag_phase(input.data(), output.data());
        return output;
    }

    /* Convenience single-precision polar-to-real inverse transform. */
    std::vector<float> inverse_mag_phase_f32(const std::vector<complex_f32>& input) const {
        if (input.size() != bins()) {
            throw error(BFFT_ERROR_INVALID_ARGUMENT);
        }
        std::vector<float> output(size());
        inverse_mag_phase_f32(input.data(), output.data());
        return output;
    }

    /* Convenience single-precision inverse transform. */
    std::vector<float> inverse_f32(const std::vector<complex_f32>& input) const {
        if (input.size() != bins()) {
            throw error(BFFT_ERROR_INVALID_ARGUMENT);
        }
        std::vector<float> output(size());
        inverse_f32(input.data(), output.data());
        return output;
    }

    /* Convenience standard inverse transform that allocates the output vector. */
    std::vector<double> inverse(const std::vector<complex>& input) const {
        if (input.size() != bins()) {
            throw error(BFFT_ERROR_INVALID_ARGUMENT);
        }
        std::vector<double> output(size());
        inverse(input.data(), output.data());
        return output;
    }

    /* Native-order inverse transform. */
    void inverse_native(const complex* input, double* output) const {
        check(bfft_inverse_native(impl_.get(), input, output));
    }

    /* Single-precision native-order inverse transform. */
    void inverse_native_f32(const complex_f32* input, float* output) const {
        check(bfft_inverse_native_f32(impl_.get(), input, output));
    }

    /* Forward transform to N residue-domain doubles. */
    void forward_residues(const double* input, double* residues) const {
        check(bfft_forward_residues(impl_.get(), input, residues));
    }

    /* In-place inverse from residue coordinates to time samples. */
    void inverse_residues(double* residues_signal) const {
        check(bfft_inverse_residues(impl_.get(), residues_signal));
    }

    /* Convert native-order complex bins to standard FFT-order bins. */
    void native_to_standard(const complex* native_input, complex* standard_output) const {
        check(bfft_native_to_standard(impl_.get(), native_input, standard_output));
    }

    /* Convert standard FFT-order bins to native-order complex bins. */
    void standard_to_native(const complex* standard_input, complex* native_output) const {
        check(bfft_standard_to_native(impl_.get(), standard_input, native_output));
    }

    /* Convert native-order float32 bins to standard FFT-order bins. */
    void native_to_standard_f32(const complex_f32* native_input, complex_f32* standard_output) const {
        check(bfft_native_to_standard_f32(impl_.get(), native_input, standard_output));
    }

    /* Convert standard FFT-order float32 bins to native-order bins. */
    void standard_to_native_f32(const complex_f32* standard_input, complex_f32* native_output) const {
        check(bfft_standard_to_native_f32(impl_.get(), standard_input, native_output));
    }

    /* Number of doubles in a residue-domain filter. */
    std::size_t filter_size() const noexcept {
        return bfft_filter_size(impl_.get());
    }

    /* Convert a standard complex response to a residue-domain filter. */
    void residue_filter_from_standard(const complex* response, double* residue_filter) const {
        check(bfft_residue_filter_from_standard(impl_.get(), response, residue_filter));
    }

    /* Convert a real zero-phase response to a residue-domain filter. */
    void residue_filter_from_real(const double* response, double* residue_filter) const {
        check(bfft_residue_filter_from_real(impl_.get(), response, residue_filter));
    }

    /* Apply a residue-domain filter in place to residue coordinates. */
    void apply_residue_filter(double* residues, const double* residue_filter) const {
        check(bfft_apply_residue_filter(impl_.get(), residues, residue_filter));
    }

    /* Filter a time-domain signal through a residue-domain filter. */
    void filter_signal(const double* input, const double* residue_filter, double* output) const {
        check(bfft_filter_signal(impl_.get(), input, residue_filter, output));
    }

private:
    struct deleter {
        void operator()(bfft_plan* plan_ptr) const noexcept {
            bfft_plan_destroy(plan_ptr);
        }
    };

    static void check(status result) {
        if (result != BFFT_OK) {
            throw error(result);
        }
    }

    std::unique_ptr<bfft_plan, deleter> impl_;
};

} // namespace bfft

#endif
