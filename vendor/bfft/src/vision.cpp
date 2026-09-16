#include <bfft/vision.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <thread>
#include <vector>

namespace {

bool checked_product(std::size_t a, std::size_t b, std::size_t* result) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        return false;
    }
    *result = a * b;
    return true;
}

bool valid_cell(std::int32_t index, std::size_t cell_count) {
    return index >= 0 && static_cast<std::size_t>(index) < cell_count;
}

bool valid_slot(std::int64_t index, std::size_t block_count) {
    return index >= 0 && static_cast<std::size_t>(index) < block_count;
}

template <typename Function>
void parallel_ranges(
    std::size_t tasks,
    std::size_t work_per_task,
    std::size_t requested_threads,
    const Function& function) {
    if (tasks == 0) {
        return;
    }
    std::size_t workers = requested_threads;
    if (workers == 0) {
        workers = std::max(
            std::size_t{1},
            static_cast<std::size_t>(std::thread::hardware_concurrency()));
    }
    workers = std::min(workers, tasks);
    workers = std::min<std::size_t>(workers, 16);
    std::size_t work = 0;
    if (!checked_product(tasks, work_per_task, &work) || work < 65536) {
        workers = 1;
    }
    if (workers == 1) {
        function(0, tasks);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (std::size_t worker = 0; worker < workers; ++worker) {
        const std::size_t begin = tasks * worker / workers;
        const std::size_t end = tasks * (worker + 1) / workers;
        pool.emplace_back(function, begin, end);
    }
    for (auto& worker : pool) {
        worker.join();
    }
}

std::size_t reflected_index(
    std::ptrdiff_t index,
    std::size_t size,
    bool mirror_without_edge) {
    if (size <= 1) {
        return 0;
    }
    const std::ptrdiff_t extent = static_cast<std::ptrdiff_t>(size);
    while (index < 0 || index >= extent) {
        if (index < 0) {
            index = mirror_without_edge ? -index : -index - 1;
        } else {
            index = mirror_without_edge
                ? 2 * extent - index - 2
                : 2 * extent - index - 1;
        }
    }
    return static_cast<std::size_t>(index);
}

}  // namespace

extern "C" {

bfft_status bfft_vision_assemble_normal(
    std::size_t pixel_count,
    std::size_t cell_count,
    std::size_t basis_width,
    std::size_t normal_block_count,
    const std::int32_t* owner,
    const std::int32_t* runner,
    const std::uint8_t* has_runner,
    const double* owner_weight,
    const double* runner_weight,
    const double* owner_basis,
    const double* runner_basis,
    const double* target,
    const std::int64_t* diagonal_block,
    const std::int64_t* owner_runner_block,
    const std::int64_t* runner_owner_block,
    double* normal_blocks,
    double* rhs) {
    if (pixel_count == 0 || cell_count == 0 || basis_width == 0 ||
        normal_block_count == 0 || owner == nullptr || runner == nullptr ||
        has_runner == nullptr || owner_weight == nullptr ||
        runner_weight == nullptr || owner_basis == nullptr ||
        runner_basis == nullptr || target == nullptr ||
        diagonal_block == nullptr || owner_runner_block == nullptr ||
        runner_owner_block == nullptr || normal_blocks == nullptr ||
        rhs == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }

    std::size_t block_area = 0;
    std::size_t normal_size = 0;
    std::size_t rhs_width = 0;
    std::size_t rhs_size = 0;
    if (!checked_product(basis_width, basis_width, &block_area) ||
        !checked_product(normal_block_count, block_area, &normal_size) ||
        !checked_product(basis_width, std::size_t{3}, &rhs_width) ||
        !checked_product(cell_count, rhs_width, &rhs_size)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }

    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        if (!valid_slot(diagonal_block[cell], normal_block_count)) {
            return BFFT_ERROR_INVALID_ARGUMENT;
        }
    }
    for (std::size_t p = 0; p < pixel_count; ++p) {
        if (!valid_cell(owner[p], cell_count) ||
            !valid_cell(runner[p], cell_count)) {
            return BFFT_ERROR_INVALID_ARGUMENT;
        }
        if (has_runner[p] != 0 &&
            (!valid_slot(owner_runner_block[p], normal_block_count) ||
             !valid_slot(runner_owner_block[p], normal_block_count))) {
            return BFFT_ERROR_INVALID_ARGUMENT;
        }
    }

    std::fill(normal_blocks, normal_blocks + normal_size, 0.0);
    std::fill(rhs, rhs + rhs_size, 0.0);

    for (std::size_t p = 0; p < pixel_count; ++p) {
        const std::size_t i = static_cast<std::size_t>(owner[p]);
        const double* first = owner_basis + p * basis_width;
        const double* target_p = target + p * 3;
        const double wi = owner_weight[p];
        double* diagonal_i =
            normal_blocks +
            static_cast<std::size_t>(diagonal_block[i]) * block_area;
        double* rhs_i = rhs + i * rhs_width;

        for (std::size_t a = 0; a < basis_width; ++a) {
            const double ua = wi * first[a];
            double* diagonal_row = diagonal_i + a * basis_width;
            for (std::size_t b = 0; b < basis_width; ++b) {
                diagonal_row[b] += ua * (wi * first[b]);
            }
            double* rhs_row = rhs_i + a * 3;
            rhs_row[0] += ua * target_p[0];
            rhs_row[1] += ua * target_p[1];
            rhs_row[2] += ua * target_p[2];
        }

        if (has_runner[p] == 0) {
            continue;
        }

        const std::size_t j = static_cast<std::size_t>(runner[p]);
        const double* second = runner_basis + p * basis_width;
        const double wj = runner_weight[p];
        double* diagonal_j =
            normal_blocks +
            static_cast<std::size_t>(diagonal_block[j]) * block_area;
        double* cross_ij =
            normal_blocks +
            static_cast<std::size_t>(owner_runner_block[p]) * block_area;
        double* cross_ji =
            normal_blocks +
            static_cast<std::size_t>(runner_owner_block[p]) * block_area;
        double* rhs_j = rhs + j * rhs_width;

        for (std::size_t a = 0; a < basis_width; ++a) {
            const double ua = wi * first[a];
            const double va = wj * second[a];
            double* diagonal_row = diagonal_j + a * basis_width;
            double* cross_ij_row = cross_ij + a * basis_width;
            double* cross_ji_row = cross_ji + a * basis_width;
            for (std::size_t b = 0; b < basis_width; ++b) {
                const double ub = wi * first[b];
                const double vb = wj * second[b];
                diagonal_row[b] += va * vb;
                cross_ij_row[b] += ua * vb;
                cross_ji_row[b] += va * ub;
            }
            double* rhs_row = rhs_j + a * 3;
            rhs_row[0] += va * target_p[0];
            rhs_row[1] += va * target_p[1];
            rhs_row[2] += va * target_p[2];
        }
    }
    return BFFT_OK;
}

bfft_status bfft_vision_render_affine(
    std::size_t pixel_count,
    std::size_t cell_count,
    std::size_t basis_width,
    const std::int32_t* owner,
    const std::int32_t* runner,
    const double* owner_weight,
    const double* runner_weight,
    const double* owner_basis,
    const double* runner_basis,
    const double* coeff,
    double* owner_prediction,
    double* runner_prediction,
    double* field) {
    if (pixel_count == 0 || cell_count == 0 || basis_width == 0 ||
        owner == nullptr || runner == nullptr || owner_weight == nullptr ||
        runner_weight == nullptr || owner_basis == nullptr ||
        runner_basis == nullptr || coeff == nullptr ||
        owner_prediction == nullptr || runner_prediction == nullptr ||
        field == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    for (std::size_t p = 0; p < pixel_count; ++p) {
        if (!valid_cell(owner[p], cell_count) ||
            !valid_cell(runner[p], cell_count)) {
            return BFFT_ERROR_INVALID_ARGUMENT;
        }
    }

    std::size_t coefficient_stride = 0;
    if (!checked_product(basis_width, std::size_t{3},
                         &coefficient_stride)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }

    for (std::size_t p = 0; p < pixel_count; ++p) {
        const std::size_t i = static_cast<std::size_t>(owner[p]);
        const std::size_t j = static_cast<std::size_t>(runner[p]);
        const double* first = owner_basis + p * basis_width;
        const double* second = runner_basis + p * basis_width;
        const double* coeff_i = coeff + i * coefficient_stride;
        const double* coeff_j = coeff + j * coefficient_stride;
        double* pred_i = owner_prediction + p * 3;
        double* pred_j = runner_prediction + p * 3;
        double* result = field + p * 3;

        for (std::size_t channel = 0; channel < 3; ++channel) {
            double left = 0.0;
            double right = 0.0;
            for (std::size_t a = 0; a < basis_width; ++a) {
                left += coeff_i[a * 3 + channel] * first[a];
                right += coeff_j[a * 3 + channel] * second[a];
            }
            pred_i[channel] = left;
            pred_j[channel] = right;
            result[channel] =
                owner_weight[p] * left + runner_weight[p] * right;
        }
    }
    return BFFT_OK;
}

bfft_status bfft_vision_scan_residual_ridges(
    std::size_t pixel_count,
    std::size_t cell_count,
    std::size_t angle_count,
    std::size_t bin_count,
    double spacing,
    double span,
    const std::int32_t* owner,
    const double* pixel_weight,
    const double* residual,
    const double* dx,
    const double* dy,
    const double* angle_cos,
    const double* angle_sin,
    const double* channel_weight,
    double* score,
    std::int32_t* best_angle,
    std::int32_t* best_bin) {
    if (pixel_count == 0 || cell_count == 0 || angle_count == 0 ||
        bin_count == 0 || !std::isfinite(spacing) || spacing <= 0.0 ||
        !std::isfinite(span) || span <= 0.0 ||
        angle_count >
            static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        bin_count >
            static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        owner == nullptr || pixel_weight == nullptr || residual == nullptr ||
        dx == nullptr || dy == nullptr || angle_cos == nullptr ||
        angle_sin == nullptr || channel_weight == nullptr || score == nullptr ||
        best_angle == nullptr || best_bin == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    for (std::size_t p = 0; p < pixel_count; ++p) {
        if (!valid_cell(owner[p], cell_count)) {
            return BFFT_ERROR_INVALID_ARGUMENT;
        }
    }

    std::size_t angle_bins = 0;
    std::size_t cell_angle_bins = 0;
    std::size_t accumulator_size = 0;
    if (!checked_product(angle_count, bin_count, &angle_bins) ||
        !checked_product(cell_count, angle_bins, &cell_angle_bins) ||
        !checked_product(cell_angle_bins, std::size_t{3},
                         &accumulator_size)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }

    try {
        std::vector<double> accumulator(accumulator_size, 0.0);
        std::vector<double> mass(cell_count, 0.0);
        std::vector<double> total(cell_count * 3, 0.0);
        const double scale = static_cast<double>(bin_count) / (2.0 * span);

        for (std::size_t p = 0; p < pixel_count; ++p) {
            const std::size_t cell = static_cast<std::size_t>(owner[p]);
            const double phi = pixel_weight[p];
            const double r0 = phi * residual[p * 3];
            const double r1 = phi * residual[p * 3 + 1];
            const double r2 = phi * residual[p * 3 + 2];
            mass[cell] += phi;
            total[cell * 3] += r0;
            total[cell * 3 + 1] += r1;
            total[cell * 3 + 2] += r2;

            const double px = dx[p] / spacing;
            const double py = dy[p] / spacing;
            const std::size_t cell_base = cell * angle_bins * 3;
            for (std::size_t angle = 0; angle < angle_count; ++angle) {
                const double projection =
                    px * angle_cos[angle] + py * angle_sin[angle];
                std::int64_t bin = static_cast<std::int64_t>(
                    (projection + span) * scale);
                bin = std::max<std::int64_t>(0, bin);
                bin = std::min<std::int64_t>(
                    static_cast<std::int64_t>(bin_count - 1), bin);
                const std::size_t slot =
                    cell_base + (angle * bin_count +
                                 static_cast<std::size_t>(bin)) * 3;
                accumulator[slot] += r0;
                accumulator[slot + 1] += r1;
                accumulator[slot + 2] += r2;
            }
        }

        for (std::size_t cell = 0; cell < cell_count; ++cell) {
            const double cell_mass = std::max(mass[cell], 1e-9);
            const double total0 = total[cell * 3];
            const double total1 = total[cell * 3 + 1];
            const double total2 = total[cell * 3 + 2];
            const std::size_t cell_base = cell * angle_bins * 3;
            bool seen = false;
            double top = 0.0;
            std::size_t top_angle = 0;
            std::size_t top_bin = 0;

            for (std::size_t angle = 0; angle < angle_count; ++angle) {
                double run0 = 0.0;
                double run1 = 0.0;
                double run2 = 0.0;
                double local_top = 0.0;
                std::size_t local_bin = 0;
                for (std::size_t bin = 0; bin < bin_count; ++bin) {
                    const std::size_t slot =
                        cell_base + (angle * bin_count + bin) * 3;
                    run0 += accumulator[slot];
                    run1 += accumulator[slot + 1];
                    run2 += accumulator[slot + 2];
                    const double c0 = total0 - 2.0 * run0;
                    const double c1 = total1 - 2.0 * run1;
                    const double c2 = total2 - 2.0 * run2;
                    const double value =
                        (channel_weight[0] * c0 * c0 +
                         channel_weight[1] * c1 * c1 +
                         channel_weight[2] * c2 * c2) /
                        cell_mass;
                    if (bin == 0 || value > local_top) {
                        local_top = value;
                        local_bin = bin;
                    }
                }
                if (!seen || local_top > top) {
                    seen = true;
                    top = local_top;
                    top_angle = angle;
                    top_bin = local_bin;
                }
            }
            score[cell] = top;
            best_angle[cell] = static_cast<std::int32_t>(top_angle);
            best_bin[cell] = static_cast<std::int32_t>(top_bin);
        }
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
    return BFFT_OK;
}

bfft_status bfft_vision_scan_paired_offsets(
    std::size_t pixel_count,
    std::size_t cell_count,
    std::size_t bin_count,
    double span,
    const std::int32_t* owner,
    const double* pixel_weight,
    const double* residual,
    const double* projection,
    const double* channel_weight,
    double* score,
    std::int32_t* best_bin) {
    if (pixel_count == 0 || cell_count == 0 || bin_count == 0 ||
        !std::isfinite(span) || span <= 0.0 ||
        bin_count >
            static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        owner == nullptr || pixel_weight == nullptr || residual == nullptr ||
        projection == nullptr || channel_weight == nullptr || score == nullptr ||
        best_bin == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }

    try {
        std::vector<std::size_t> offset(cell_count + 1, 0);
        for (std::size_t p = 0; p < pixel_count; ++p) {
            if (!valid_cell(owner[p], cell_count) ||
                !std::isfinite(pixel_weight[p]) ||
                !std::isfinite(projection[p])) {
                return BFFT_ERROR_INVALID_ARGUMENT;
            }
            ++offset[static_cast<std::size_t>(owner[p]) + 1];
        }
        for (std::size_t cell = 0; cell < cell_count; ++cell) {
            offset[cell + 1] += offset[cell];
        }
        std::vector<std::size_t> cursor(offset.begin(), offset.end() - 1);
        std::vector<std::size_t> order(pixel_count);
        for (std::size_t p = 0; p < pixel_count; ++p) {
            const std::size_t cell = static_cast<std::size_t>(owner[p]);
            order[cursor[cell]++] = p;
        }

        std::size_t histogram_size = 0;
        if (!checked_product(bin_count, std::size_t{3}, &histogram_size)) {
            return BFFT_ERROR_INVALID_ARGUMENT;
        }
        const double scale = static_cast<double>(bin_count) / (2.0 * span);
        const auto scan_cells = [&] (
            std::size_t begin, std::size_t end) {
            std::vector<double> histogram(histogram_size, 0.0);
            for (std::size_t cell = begin; cell < end; ++cell) {
                std::fill(histogram.begin(), histogram.end(), 0.0);
                double mass = 0.0;
                double total0 = 0.0;
                double total1 = 0.0;
                double total2 = 0.0;
                for (std::size_t at = offset[cell];
                     at < offset[cell + 1]; ++at) {
                    const std::size_t p = order[at];
                    const double phi = pixel_weight[p];
                    const double r0 = phi * residual[p * 3];
                    const double r1 = phi * residual[p * 3 + 1];
                    const double r2 = phi * residual[p * 3 + 2];
                    mass += phi;
                    total0 += r0;
                    total1 += r1;
                    total2 += r2;
                    std::int64_t bin = static_cast<std::int64_t>(
                        (projection[p] + span) * scale);
                    bin = std::max<std::int64_t>(0, bin);
                    bin = std::min<std::int64_t>(
                        static_cast<std::int64_t>(bin_count - 1), bin);
                    const std::size_t slot =
                        static_cast<std::size_t>(bin) * 3;
                    histogram[slot] += r0;
                    histogram[slot + 1] += r1;
                    histogram[slot + 2] += r2;
                }

                const double denominator = std::max(mass, 1e-9);
                double run0 = 0.0;
                double run1 = 0.0;
                double run2 = 0.0;
                double top = 0.0;
                std::size_t top_bin = 0;
                bool seen = false;
                for (std::size_t bin = 0; bin < bin_count; ++bin) {
                    const std::size_t slot = bin * 3;
                    run0 += histogram[slot];
                    run1 += histogram[slot + 1];
                    run2 += histogram[slot + 2];
                    const double contrast0 = total0 - 2.0 * run0;
                    const double contrast1 = total1 - 2.0 * run1;
                    const double contrast2 = total2 - 2.0 * run2;
                    const double value =
                        (channel_weight[0] * contrast0 * contrast0 +
                         channel_weight[1] * contrast1 * contrast1 +
                         channel_weight[2] * contrast2 * contrast2) /
                        denominator;
                    if (!seen || value > top) {
                        top = value;
                        top_bin = bin;
                        seen = true;
                    }
                }
                score[cell] = top;
                best_bin[cell] = static_cast<std::int32_t>(top_bin);
            }
        };
        const std::size_t mean_pixels =
            pixel_count / std::max(cell_count, std::size_t{1}) + 1;
        parallel_ranges(
            cell_count,
            histogram_size + mean_pixels * 4,
            0,
            scan_cells);
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
    }
    return BFFT_OK;
}

bfft_status bfft_vision_support_forward(
    std::size_t sample_count, std::size_t pixel_count,
    std::size_t cell_count, const std::int32_t* rows,
    const std::int32_t* sites, const double* weight,
    const double* basis_x, const double* basis_y,
    const double* coefficient, double* pixel) {
    if (sample_count == 0 || pixel_count == 0 || cell_count == 0 ||
        rows == nullptr || sites == nullptr || weight == nullptr ||
        basis_x == nullptr || basis_y == nullptr || coefficient == nullptr ||
        pixel == nullptr)
        return BFFT_ERROR_INVALID_ARGUMENT;
    std::fill(pixel, pixel + pixel_count, 0.0);
    for (std::size_t sample = 0; sample < sample_count; ++sample) {
        const std::int32_t row = rows[sample];
        const std::int32_t site = sites[sample];
        if (row < 0 || static_cast<std::size_t>(row) >= pixel_count ||
            !valid_cell(site, cell_count))
            return BFFT_ERROR_INVALID_ARGUMENT;
        const double* c =
            coefficient + 3 * static_cast<std::size_t>(site);
        pixel[static_cast<std::size_t>(row)] += weight[sample] * (
            c[0] + basis_x[sample] * c[1] + basis_y[sample] * c[2]);
    }
    return BFFT_OK;
}

bfft_status bfft_vision_support_transpose(
    std::size_t sample_count, std::size_t pixel_count,
    std::size_t cell_count, const std::int32_t* rows,
    const std::int32_t* sites, const double* weight,
    const double* basis_x, const double* basis_y, const double* pixel,
    double* coefficient) {
    if (sample_count == 0 || pixel_count == 0 || cell_count == 0 ||
        rows == nullptr || sites == nullptr || weight == nullptr ||
        basis_x == nullptr || basis_y == nullptr || pixel == nullptr ||
        coefficient == nullptr)
        return BFFT_ERROR_INVALID_ARGUMENT;
    std::fill(coefficient, coefficient + 3 * cell_count, 0.0);
    for (std::size_t sample = 0; sample < sample_count; ++sample) {
        const std::int32_t row = rows[sample];
        const std::int32_t site = sites[sample];
        if (row < 0 || static_cast<std::size_t>(row) >= pixel_count ||
            !valid_cell(site, cell_count))
            return BFFT_ERROR_INVALID_ARGUMENT;
        const double value =
            weight[sample] * pixel[static_cast<std::size_t>(row)];
        double* c = coefficient + 3 * static_cast<std::size_t>(site);
        c[0] += value;
        c[1] += value * basis_x[sample];
        c[2] += value * basis_y[sample];
    }
    return BFFT_OK;
}

bfft_status bfft_vision_support_normal_apply(
    std::size_t sample_count, std::size_t pixel_count,
    std::size_t cell_count, const std::int32_t* rows,
    const std::int32_t* sites, const double* weight,
    const double* basis_x, const double* basis_y,
    const double* coefficient, double* pixel_scratch, double* output) {
    const bfft_status forward = bfft_vision_support_forward(
        sample_count, pixel_count, cell_count, rows, sites, weight,
        basis_x, basis_y, coefficient, pixel_scratch);
    if (forward != BFFT_OK) return forward;
    return bfft_vision_support_transpose(
        sample_count, pixel_count, cell_count, rows, sites, weight,
        basis_x, basis_y, pixel_scratch, output);
}

bfft_status bfft_vision_curvature_population_f32(
    std::size_t height, std::size_t width,
    const float* precision_xx, const float* precision_xy,
    const float* precision_yy, const float* base_measure,
    double base_implied_cells,
    float* corrected_measure, float* director_curvature,
    float* sagitta_ratio, float* population_factor,
    double* corrected_implied_cells) {
    if (height < 2 || width < 2 ||
        precision_xx == nullptr || precision_xy == nullptr ||
        precision_yy == nullptr || base_measure == nullptr ||
        corrected_measure == nullptr || director_curvature == nullptr ||
        sagitta_ratio == nullptr || population_factor == nullptr ||
        corrected_implied_cells == nullptr ||
        !std::isfinite(base_implied_cells) || base_implied_cells <= 0.0) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    std::size_t pixels = 0;
    if (!checked_product(height, width, &pixels)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }

    try {
        std::vector<double> doubled_cosine(pixels);
        std::vector<double> doubled_sine(pixels);
        std::vector<double> corrected_raw(pixels);
        for (std::size_t p = 0; p < pixels; ++p) {
            const double qxx = static_cast<double>(precision_xx[p]);
            const double qxy = static_cast<double>(precision_xy[p]);
            const double qyy = static_cast<double>(precision_yy[p]);
            const double measure = static_cast<double>(base_measure[p]);
            if (!std::isfinite(qxx) || !std::isfinite(qxy) ||
                !std::isfinite(qyy) || !std::isfinite(measure) ||
                measure < 0.0) {
                return BFFT_ERROR_INVALID_ARGUMENT;
            }
            const double discriminant =
                std::hypot(qxx - qyy, 2.0 * qxy);
            const double safe = std::max(discriminant, 1e-30);
            doubled_cosine[p] = (qxx - qyy) / safe;
            doubled_sine[p] = 2.0 * qxy / safe;
        }

        const auto difference_x = [width](
            const std::vector<double>& field,
            std::size_t y, std::size_t x) {
            const std::size_t p = y * width + x;
            if (x == 0) return field[p + 1] - field[p];
            if (x + 1 == width) return field[p] - field[p - 1];
            return 0.5 * (field[p + 1] - field[p - 1]);
        };
        const auto difference_y = [height, width](
            const std::vector<double>& field,
            std::size_t y, std::size_t x) {
            const std::size_t p = y * width + x;
            if (y == 0) return field[p + width] - field[p];
            if (y + 1 == height) return field[p] - field[p - width];
            return 0.5 * (field[p + width] - field[p - width]);
        };

        double implied = 0.0;
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const std::size_t p = y * width + x;
                const double qxx = static_cast<double>(precision_xx[p]);
                const double qxy = static_cast<double>(precision_xy[p]);
                const double qyy = static_cast<double>(precision_yy[p]);
                const double trace = qxx + qyy;
                const double discriminant =
                    std::hypot(qxx - qyy, 2.0 * qxy);
                const double coherence =
                    discriminant / std::max(trace, 1e-30);
                const double u = doubled_cosine[p];
                const double v = doubled_sine[p];
                const double theta_x = 0.5 * (
                    u * difference_x(doubled_sine, y, x) -
                    v * difference_x(doubled_cosine, y, x));
                const double theta_y = 0.5 * (
                    u * difference_y(doubled_sine, y, x) -
                    v * difference_y(doubled_cosine, y, x));
                const double normal_x =
                    std::sqrt(std::max(0.5 * (1.0 + u), 0.0));
                const double sign_source =
                    std::abs(v) > 1e-30 ? v : 1.0;
                const double normal_y = std::copysign(
                    std::sqrt(std::max(0.5 * (1.0 - u), 0.0)),
                    sign_source);
                const double tangent_x = -normal_y;
                const double tangent_y = normal_x;
                const double curvature = coherence * std::abs(
                    tangent_x * theta_x + tangent_y * theta_y);
                const double high = std::max(
                    0.5 * (trace + discriminant), 1e-30);
                const double low = std::max(
                    0.5 * (trace - discriminant), 1e-30);
                const double tangent_span = 1.0 / std::sqrt(low);
                const double normal_span = 1.0 / std::sqrt(high);
                const double ratio =
                    curvature * tangent_span * tangent_span /
                    std::max(2.0 * normal_span, 1e-30);
                const double factor = std::sqrt(std::max(1.0, ratio));
                const double raw =
                    static_cast<double>(base_measure[p]) *
                    base_implied_cells * factor;
                corrected_raw[p] = raw;
                implied += raw;
                director_curvature[p] = static_cast<float>(curvature);
                sagitta_ratio[p] = static_cast<float>(ratio);
                population_factor[p] = static_cast<float>(factor);
            }
        }
        if (!std::isfinite(implied) || implied <= 0.0) {
            return BFFT_ERROR_INVALID_ARGUMENT;
        }
        const double inverse_implied = 1.0 / implied;
        for (std::size_t p = 0; p < pixels; ++p) {
            corrected_measure[p] = static_cast<float>(
                corrected_raw[p] * inverse_implied);
        }
        *corrected_implied_cells = implied;
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
    return BFFT_OK;
}

bfft_status bfft_vision_soft_support_diffuse(
    std::size_t height, std::size_t width, std::size_t channels,
    std::size_t passes, std::size_t thread_count, double coupling,
    const double* field,
    const double* horizontal, const double* vertical,
    const double* diagonal_down_right, const double* diagonal_down_left,
    double* output, double* scratch) {
    if (height < 2 || width < 2 || channels == 0 ||
        !std::isfinite(coupling) || coupling < 0.0 ||
        field == nullptr || horizontal == nullptr || vertical == nullptr ||
        diagonal_down_right == nullptr || diagonal_down_left == nullptr ||
        output == nullptr || scratch == nullptr || output == scratch) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    std::size_t pixels = 0;
    std::size_t values = 0;
    if (!checked_product(height, width, &pixels) ||
        !checked_product(pixels, channels, &values)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    std::copy(field, field + values, output);
    if (passes == 0 || coupling == 0.0) {
        return BFFT_OK;
    }

    try {
    const std::size_t horizontal_stride = width - 1;
    std::vector<double> inverse_denominator(pixels);
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            double denominator = 1.0;
            if (x > 0)
                denominator += coupling *
                    horizontal[y * horizontal_stride + x - 1];
            if (x + 1 < width)
                denominator += coupling *
                    horizontal[y * horizontal_stride + x];
            if (y > 0)
                denominator += coupling * vertical[(y - 1) * width + x];
            if (y + 1 < height)
                denominator += coupling * vertical[y * width + x];
            if (x > 0 && y > 0)
                denominator += coupling * diagonal_down_right[
                    (y - 1) * horizontal_stride + x - 1];
            if (x + 1 < width && y + 1 < height)
                denominator += coupling * diagonal_down_right[
                    y * horizontal_stride + x];
            if (x + 1 < width && y > 0)
                denominator += coupling * diagonal_down_left[
                    (y - 1) * horizontal_stride + x];
            if (x > 0 && y + 1 < height)
                denominator += coupling * diagonal_down_left[
                    y * horizontal_stride + x - 1];
            inverse_denominator[y * width + x] = 1.0 / denominator;
        }
    }

    double* source = output;
    double* target = scratch;
    std::size_t workers = thread_count == 0 ? 1 : thread_count;
    workers = std::min(workers, height);
    workers = std::min<std::size_t>(workers, 64);
    const auto process_rows = [&](std::size_t y_begin, std::size_t y_end) {
        for (std::size_t y = y_begin; y < y_end; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const std::size_t p = y * width + x;
                const std::size_t base = p * channels;
                for (std::size_t channel = 0;
                     channel < channels; ++channel) {
                    double numerator = source[base + channel];
                    if (x > 0)
                        numerator += coupling *
                            horizontal[y * horizontal_stride + x - 1] *
                            source[(p - 1) * channels + channel];
                    if (x + 1 < width)
                        numerator += coupling *
                            horizontal[y * horizontal_stride + x] *
                            source[(p + 1) * channels + channel];
                    if (y > 0)
                        numerator += coupling *
                            vertical[(y - 1) * width + x] *
                            source[(p - width) * channels + channel];
                    if (y + 1 < height)
                        numerator += coupling *
                            vertical[y * width + x] *
                            source[(p + width) * channels + channel];
                    if (x > 0 && y > 0)
                        numerator += coupling * diagonal_down_right[
                            (y - 1) * horizontal_stride + x - 1] *
                            source[(p - width - 1) * channels + channel];
                    if (x + 1 < width && y + 1 < height)
                        numerator += coupling * diagonal_down_right[
                            y * horizontal_stride + x] *
                            source[(p + width + 1) * channels + channel];
                    if (x + 1 < width && y > 0)
                        numerator += coupling * diagonal_down_left[
                            (y - 1) * horizontal_stride + x] *
                            source[(p - width + 1) * channels + channel];
                    if (x > 0 && y + 1 < height)
                        numerator += coupling * diagonal_down_left[
                            y * horizontal_stride + x - 1] *
                            source[(p + width - 1) * channels + channel];
                    target[base + channel] =
                        numerator * inverse_denominator[p];
                }
            }
        }
    };
    for (std::size_t pass = 0; pass < passes; ++pass) {
        if (workers == 1) {
            process_rows(0, height);
        } else {
            std::vector<std::thread> pool;
            pool.reserve(workers);
            for (std::size_t worker = 0; worker < workers; ++worker) {
                const std::size_t y_begin = height * worker / workers;
                const std::size_t y_end = height * (worker + 1) / workers;
                pool.emplace_back(process_rows, y_begin, y_end);
            }
            for (auto& worker : pool) {
                worker.join();
            }
        }
        std::swap(source, target);
    }
    if (source != output) {
        std::copy(source, source + values, output);
    }
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
    return BFFT_OK;
}

bfft_status bfft_vision_prepare_continuous_metric(
    std::size_t height,
    std::size_t width,
    double consistency_limit,
    const std::int32_t* superbase,
    const double* mxx,
    const double* mxy,
    const double* myy,
    std::int32_t* directions,
    double* direction_costs,
    std::uint8_t* direction_valid,
    double* cardinal_costs,
    std::int64_t* inverse_offset,
    std::size_t inverse_capacity,
    std::int32_t* inverse_receiver,
    std::size_t* inverse_count) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    std::size_t pixels = 0;
    if (height == 0 || width == 0 ||
        !checked_product(height, width, &pixels) ||
        !std::isfinite(consistency_limit) || consistency_limit < 1.0 ||
        superbase == nullptr || mxx == nullptr || mxy == nullptr ||
        myy == nullptr || directions == nullptr ||
        direction_costs == nullptr || direction_valid == nullptr ||
        cardinal_costs == nullptr || inverse_offset == nullptr ||
        inverse_receiver == nullptr || inverse_count == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    constexpr double infinity = std::numeric_limits<double>::infinity();
    static constexpr int cardinal_x[4] = {1, -1, 0, 0};
    static constexpr int cardinal_y[4] = {0, 0, 1, -1};
    const auto metric_norm = [](
        int dx, int dy, double a, double b, double c) {
        const double x = static_cast<double>(dx);
        const double y = static_cast<double>(dy);
        return std::sqrt(std::max(
            a * x * x + 2.0 * b * x * y + c * y * y,
            1e-30));
    };
    try {
        /* Expand and cyclically order each signed reduced superbase. */
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            const std::size_t source_base = pixel * 3 * 2;
            const std::size_t direction_base = pixel * 6 * 2;
            for (std::size_t index = 0; index < 3; ++index) {
                const int dx = superbase[source_base + index * 2];
                const int dy = superbase[source_base + index * 2 + 1];
                directions[direction_base + index * 2] = dx;
                directions[direction_base + index * 2 + 1] = dy;
                directions[direction_base + (index + 3) * 2] = -dx;
                directions[direction_base + (index + 3) * 2 + 1] = -dy;
            }
            for (std::size_t index = 1; index < 6; ++index) {
                const int key_x = directions[direction_base + index * 2];
                const int key_y = directions[
                    direction_base + index * 2 + 1];
                const int key_half =
                    key_y < 0 || (key_y == 0 && key_x >= 0) ? 0 : 1;
                std::size_t position = index;
                while (position > 0) {
                    const int other_x = directions[
                        direction_base + (position - 1) * 2];
                    const int other_y = directions[
                        direction_base + (position - 1) * 2 + 1];
                    const int other_half =
                        other_y < 0 || (other_y == 0 && other_x >= 0)
                        ? 0 : 1;
                    const int cross = other_x * key_y - other_y * key_x;
                    const bool comes_before =
                        key_half < other_half ||
                        (key_half == other_half && cross < 0);
                    if (!comes_before) {
                        break;
                    }
                    directions[direction_base + position * 2] = other_x;
                    directions[direction_base + position * 2 + 1] = other_y;
                    --position;
                }
                directions[direction_base + position * 2] = key_x;
                directions[direction_base + position * 2 + 1] = key_y;
            }
        }

        /* Integrate each reduced segment and the cardinal connectivity floor. */
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const std::size_t pixel = y * width + x;
                const std::size_t direction_base = pixel * 6;
                for (std::size_t index = 0; index < 6; ++index) {
                    direction_costs[direction_base + index] = infinity;
                    direction_valid[direction_base + index] = 0;
                    const int dx = directions[
                        (direction_base + index) * 2];
                    const int dy = directions[
                        (direction_base + index) * 2 + 1];
                    const std::ptrdiff_t target_x =
                        static_cast<std::ptrdiff_t>(x) + dx;
                    const std::ptrdiff_t target_y =
                        static_cast<std::ptrdiff_t>(y) + dy;
                    if (target_x < 0 || target_y < 0 ||
                        target_x >= static_cast<std::ptrdiff_t>(width) ||
                        target_y >= static_cast<std::ptrdiff_t>(height)) {
                        continue;
                    }
                    const int steps = std::max(std::abs(dx), std::abs(dy));
                    double total = 0.0;
                    double total_weight = 0.0;
                    int previous_x = 1 << 30;
                    int previous_y = 1 << 30;
                    for (int step = 0; step <= steps; ++step) {
                        const int offset_x = static_cast<int>(std::nearbyint(
                            static_cast<double>(step * dx) / steps));
                        const int offset_y = static_cast<int>(std::nearbyint(
                            static_cast<double>(step * dy) / steps));
                        if (offset_x == previous_x &&
                            offset_y == previous_y) {
                            continue;
                        }
                        previous_x = offset_x;
                        previous_y = offset_y;
                        const double weight =
                            step == 0 || step == steps ? 0.5 : 1.0;
                        const std::size_t sample =
                            static_cast<std::size_t>(
                                static_cast<std::ptrdiff_t>(y) + offset_y) *
                                width +
                            static_cast<std::size_t>(
                                static_cast<std::ptrdiff_t>(x) + offset_x);
                        total += weight * metric_norm(
                            dx, dy, mxx[sample], mxy[sample], myy[sample]);
                        total_weight += weight;
                    }
                    const double integrated =
                        total / std::max(total_weight, 1e-30);
                    const double local = metric_norm(
                        dx, dy, mxx[pixel], mxy[pixel], myy[pixel]);
                    const double ratio = std::max(
                        integrated / std::max(local, 1e-30),
                        local / std::max(integrated, 1e-30));
                    direction_costs[direction_base + index] = integrated;
                    direction_valid[direction_base + index] =
                        ratio <= consistency_limit ? 1 : 0;
                }
                const std::size_t cardinal_base = pixel * 4;
                for (std::size_t index = 0; index < 4; ++index) {
                    cardinal_costs[cardinal_base + index] = infinity;
                    const int dx = cardinal_x[index];
                    const int dy = cardinal_y[index];
                    const std::ptrdiff_t nx =
                        static_cast<std::ptrdiff_t>(x) + dx;
                    const std::ptrdiff_t ny =
                        static_cast<std::ptrdiff_t>(y) + dy;
                    if (nx < 0 || ny < 0 ||
                        nx >= static_cast<std::ptrdiff_t>(width) ||
                        ny >= static_cast<std::ptrdiff_t>(height)) {
                        continue;
                    }
                    const std::size_t neighbour =
                        static_cast<std::size_t>(ny) * width +
                        static_cast<std::size_t>(nx);
                    cardinal_costs[cardinal_base + index] = metric_norm(
                        dx,
                        dy,
                        0.5 * (mxx[pixel] + mxx[neighbour]),
                        0.5 * (mxy[pixel] + mxy[neighbour]),
                        0.5 * (myy[pixel] + myy[neighbour]));
                }
            }
        }

        /* Build the exact stable inverse-incidence CSR in two passes. */
        std::vector<std::int64_t> count(pixels, 0);
        for (std::size_t receiver = 0; receiver < pixels; ++receiver) {
            const std::size_t y = receiver / width;
            const std::size_t x = receiver - y * width;
            const std::size_t direction_base = receiver * 6;
            for (std::size_t index = 0; index < 6; ++index) {
                const int dx = directions[(direction_base + index) * 2];
                const int dy = directions[(direction_base + index) * 2 + 1];
                const std::ptrdiff_t nx =
                    static_cast<std::ptrdiff_t>(x) + dx;
                const std::ptrdiff_t ny =
                    static_cast<std::ptrdiff_t>(y) + dy;
                if (nx >= 0 && ny >= 0 &&
                    nx < static_cast<std::ptrdiff_t>(width) &&
                    ny < static_cast<std::ptrdiff_t>(height)) {
                    ++count[static_cast<std::size_t>(ny) * width +
                            static_cast<std::size_t>(nx)];
                }
            }
        }
        for (std::size_t receiver = 0; receiver < pixels; ++receiver) {
            const std::size_t y = receiver / width;
            const std::size_t x = receiver - y * width;
            const std::size_t direction_base = receiver * 6;
            for (std::size_t index = 0; index < 4; ++index) {
                const int dx = cardinal_x[index];
                const int dy = cardinal_y[index];
                const std::ptrdiff_t nx =
                    static_cast<std::ptrdiff_t>(x) + dx;
                const std::ptrdiff_t ny =
                    static_cast<std::ptrdiff_t>(y) + dy;
                if (nx < 0 || ny < 0 ||
                    nx >= static_cast<std::ptrdiff_t>(width) ||
                    ny >= static_cast<std::ptrdiff_t>(height)) {
                    continue;
                }
                bool duplicate = false;
                for (std::size_t local = 0; local < 6; ++local) {
                    duplicate = duplicate || (
                        directions[(direction_base + local) * 2] == dx &&
                        directions[(direction_base + local) * 2 + 1] == dy);
                }
                if (!duplicate) {
                    ++count[static_cast<std::size_t>(ny) * width +
                            static_cast<std::size_t>(nx)];
                }
            }
        }
        inverse_offset[0] = 0;
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            inverse_offset[pixel + 1] = inverse_offset[pixel] + count[pixel];
        }
        const std::size_t incidences = static_cast<std::size_t>(
            inverse_offset[pixels]);
        if (incidences > inverse_capacity) {
            return BFFT_ERROR_INVALID_ARGUMENT;
        }
        std::vector<std::int64_t> cursor(
            inverse_offset, inverse_offset + pixels);
        for (std::size_t receiver = 0; receiver < pixels; ++receiver) {
            const std::size_t y = receiver / width;
            const std::size_t x = receiver - y * width;
            const std::size_t direction_base = receiver * 6;
            for (std::size_t index = 0; index < 6; ++index) {
                const int dx = directions[(direction_base + index) * 2];
                const int dy = directions[(direction_base + index) * 2 + 1];
                const std::ptrdiff_t nx =
                    static_cast<std::ptrdiff_t>(x) + dx;
                const std::ptrdiff_t ny =
                    static_cast<std::ptrdiff_t>(y) + dy;
                if (nx >= 0 && ny >= 0 &&
                    nx < static_cast<std::ptrdiff_t>(width) &&
                    ny < static_cast<std::ptrdiff_t>(height)) {
                    const std::size_t vertex =
                        static_cast<std::size_t>(ny) * width +
                        static_cast<std::size_t>(nx);
                    inverse_receiver[cursor[vertex]++] =
                        static_cast<std::int32_t>(receiver);
                }
            }
        }
        for (std::size_t receiver = 0; receiver < pixels; ++receiver) {
            const std::size_t y = receiver / width;
            const std::size_t x = receiver - y * width;
            const std::size_t direction_base = receiver * 6;
            for (std::size_t index = 0; index < 4; ++index) {
                const int dx = cardinal_x[index];
                const int dy = cardinal_y[index];
                const std::ptrdiff_t nx =
                    static_cast<std::ptrdiff_t>(x) + dx;
                const std::ptrdiff_t ny =
                    static_cast<std::ptrdiff_t>(y) + dy;
                if (nx < 0 || ny < 0 ||
                    nx >= static_cast<std::ptrdiff_t>(width) ||
                    ny >= static_cast<std::ptrdiff_t>(height)) {
                    continue;
                }
                bool duplicate = false;
                for (std::size_t local = 0; local < 6; ++local) {
                    duplicate = duplicate || (
                        directions[(direction_base + local) * 2] == dx &&
                        directions[(direction_base + local) * 2 + 1] == dy);
                }
                if (!duplicate) {
                    const std::size_t vertex =
                        static_cast<std::size_t>(ny) * width +
                        static_cast<std::size_t>(nx);
                    inverse_receiver[cursor[vertex]++] =
                        static_cast<std::int32_t>(receiver);
                }
            }
        }
        *inverse_count = incidences;
        return BFFT_OK;
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
}

bfft_status bfft_vision_fast_march_first_label(
    std::size_t height,
    std::size_t width,
    std::size_t seed_count,
    const std::int32_t* seed_pixel,
    const double* seed_value,
    const std::int32_t* seed_label,
    const double* seed_gradient_x,
    const double* seed_gradient_y,
    const std::int32_t* directions,
    const double* direction_costs,
    const std::uint8_t* direction_valid,
    const double* cardinal_costs,
    const std::int64_t* inverse_offset,
    std::size_t inverse_count,
    const std::int32_t* inverse_receiver,
    const double* mxx,
    const double* mxy,
    const double* myy,
    std::int32_t* owner,
    double* distance,
    double* gradient_x,
    double* gradient_y,
    double* source_gradient_x,
    double* source_gradient_y,
    std::int32_t* parent_first,
    std::int32_t* parent_second,
    double* parent_fraction,
    std::int32_t* acceptance_order,
    std::size_t* accepted_count,
    std::size_t* push_count,
    std::size_t* maximum_heap_size) {
    std::size_t pixels = 0;
    if (height == 0 || width == 0 ||
        !checked_product(height, width, &pixels) ||
        seed_pixel == nullptr || seed_value == nullptr ||
        seed_label == nullptr || seed_gradient_x == nullptr ||
        seed_gradient_y == nullptr || directions == nullptr ||
        direction_costs == nullptr || direction_valid == nullptr ||
        cardinal_costs == nullptr || inverse_offset == nullptr ||
        (inverse_count != 0 && inverse_receiver == nullptr) ||
        mxx == nullptr || mxy == nullptr || myy == nullptr ||
        owner == nullptr || distance == nullptr ||
        push_count == nullptr || maximum_heap_size == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    const bool full_output = gradient_x != nullptr;
    const bool source_output = source_gradient_x != nullptr;
    if (
        (gradient_y != nullptr) != full_output ||
        (parent_first != nullptr) != full_output ||
        (parent_second != nullptr) != full_output ||
        (parent_fraction != nullptr) != full_output ||
        (acceptance_order != nullptr) != full_output ||
        (accepted_count != nullptr) != full_output
    ) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    if ((source_gradient_y != nullptr) != source_output ||
        (source_output && !full_output)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    if (inverse_offset[0] != 0 ||
        inverse_offset[pixels] < 0 ||
        static_cast<std::size_t>(inverse_offset[pixels]) != inverse_count) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    /*
       The Python boundary constructs this CSR once and owns its shape checks.
       Do not rescan every pixel and incidence on every global walk: that
       validation alone is an image-sized pass through the hottest API.
    */
    for (std::size_t seed = 0; seed < seed_count; ++seed) {
        if (seed_pixel[seed] < 0 ||
            static_cast<std::size_t>(seed_pixel[seed]) >= pixels ||
            seed_label[seed] < 0) {
            return BFFT_ERROR_INVALID_ARGUMENT;
        }
    }

    constexpr double infinity = 1e300;
    const auto metric_norm = [](
        double dx, double dy, double a, double b, double c) {
        return std::sqrt(std::max(
            a * dx * dx + 2.0 * b * dx * dy + c * dy * dy,
            1e-30));
    };
    const auto simplex_candidate = [&metric_norm](
        double first_value,
        double second_value,
        double first_x,
        double first_y,
        double second_x,
        double second_y,
        double a,
        double b,
        double c,
        double* fraction) {
        const double delta_value = second_value - first_value;
        const double delta_x = second_x - first_x;
        const double delta_y = second_y - first_y;
        const double quadratic_a =
            a * delta_x * delta_x +
            2.0 * b * delta_x * delta_y +
            c * delta_y * delta_y;
        const double quadratic_b =
            delta_x * (a * first_x + b * first_y) +
            delta_y * (b * first_x + c * first_y);
        const double quadratic_c =
            a * first_x * first_x +
            2.0 * b * first_x * first_y +
            c * first_y * first_y;
        const double first_length =
            std::sqrt(std::max(quadratic_c, 1e-30));
        const double derivative_first =
            delta_value + quadratic_b / first_length;
        if (derivative_first >= 0.0) {
            *fraction = 0.0;
            return first_value + first_length;
        }
        const double second_quadratic =
            quadratic_a + 2.0 * quadratic_b + quadratic_c;
        const double second_length =
            std::sqrt(std::max(second_quadratic, 1e-30));
        const double derivative_second =
            delta_value +
            (quadratic_a + quadratic_b) / second_length;
        if (derivative_second <= 0.0) {
            *fraction = 1.0;
            return second_value + second_length;
        }
        const double area = std::max(
            quadratic_a * quadratic_c - quadratic_b * quadratic_b,
            0.0);
        const double causal = std::max(
            quadratic_a - delta_value * delta_value,
            1e-30);
        double t = (
            -quadratic_b -
            delta_value * std::sqrt(area / causal)
        ) / std::max(quadratic_a, 1e-30);
        t = std::min(std::max(t, 0.0), 1.0);
        const double rx = first_x + t * delta_x;
        const double ry = first_y + t * delta_y;
        *fraction = t;
        return (
            first_value +
            t * delta_value +
            metric_norm(rx, ry, a, b, c));
    };

    try {
        std::vector<double> tentative(pixels, infinity);
        std::vector<std::int32_t> tentative_label(pixels, -1);
        std::vector<double> tentative_gradient_x(pixels, 0.0);
        std::vector<double> tentative_gradient_y(pixels, 0.0);
        std::vector<double> tentative_source_gradient_x;
        std::vector<double> tentative_source_gradient_y;
        if (source_output) {
            tentative_source_gradient_x.assign(pixels, 0.0);
            tentative_source_gradient_y.assign(pixels, 0.0);
        }
        std::vector<std::int32_t> tentative_parent_first(pixels, -1);
        std::vector<std::int32_t> tentative_parent_second(pixels, -1);
        std::vector<double> tentative_parent_fraction(pixels, 0.0);
        std::vector<std::uint8_t> accepted(pixels, 0);
        std::vector<double> heap_value(pixels);
        std::vector<std::int32_t> heap_pixel(pixels);
        std::vector<std::int32_t> heap_position(pixels, -1);

        std::fill(distance, distance + pixels, infinity);
        std::fill(owner, owner + pixels, std::int32_t{-1});
        if (full_output) {
            std::fill(gradient_x, gradient_x + pixels, 0.0);
            std::fill(gradient_y, gradient_y + pixels, 0.0);
            std::fill(parent_first, parent_first + pixels, std::int32_t{-1});
            std::fill(parent_second, parent_second + pixels, std::int32_t{-1});
            std::fill(parent_fraction, parent_fraction + pixels, 0.0);
        }
        if (source_output) {
            std::fill(source_gradient_x, source_gradient_x + pixels, 0.0);
            std::fill(source_gradient_y, source_gradient_y + pixels, 0.0);
        }

        std::size_t heap_size = 0;
        std::size_t pushes = 0;
        std::size_t maximum_heap = 0;
        std::size_t acceptance_size = 0;

        const auto bubble_up = [&] (std::size_t child) {
            while (child > 0) {
                const std::size_t parent = (child - 1) / 2;
                if (heap_value[parent] <= heap_value[child]) {
                    break;
                }
                std::swap(heap_value[parent], heap_value[child]);
                std::swap(heap_pixel[parent], heap_pixel[child]);
                heap_position[static_cast<std::size_t>(
                    heap_pixel[parent])] = static_cast<std::int32_t>(parent);
                heap_position[static_cast<std::size_t>(
                    heap_pixel[child])] = static_cast<std::int32_t>(child);
                child = parent;
            }
        };

        for (std::size_t seed = 0; seed < seed_count; ++seed) {
            const std::size_t pixel =
                static_cast<std::size_t>(seed_pixel[seed]);
            const double value = seed_value[seed];
            if (value >= tentative[pixel]) {
                continue;
            }
            tentative[pixel] = value;
            tentative_label[pixel] = seed_label[seed];
            tentative_gradient_x[pixel] = seed_gradient_x[seed];
            tentative_gradient_y[pixel] = seed_gradient_y[seed];
            if (source_output) {
                tentative_source_gradient_x[pixel] = -seed_gradient_x[seed];
                tentative_source_gradient_y[pixel] = -seed_gradient_y[seed];
            }
            std::int32_t child_position = heap_position[pixel];
            if (child_position < 0) {
                child_position = static_cast<std::int32_t>(heap_size);
                heap_pixel[heap_size] = static_cast<std::int32_t>(pixel);
                heap_position[pixel] = child_position;
                ++heap_size;
            }
            const std::size_t child =
                static_cast<std::size_t>(child_position);
            heap_value[child] = value;
            ++pushes;
            maximum_heap = std::max(maximum_heap, heap_size);
            bubble_up(child);
        }

        static constexpr int cardinal_x[4] = {1, -1, 0, 0};
        static constexpr int cardinal_y[4] = {0, 0, 1, -1};
        while (heap_size > 0) {
            const double value = heap_value[0];
            const std::size_t pixel =
                static_cast<std::size_t>(heap_pixel[0]);
            --heap_size;
            heap_position[pixel] = -2;
            if (heap_size > 0) {
                heap_value[0] = heap_value[heap_size];
                heap_pixel[0] = heap_pixel[heap_size];
                heap_position[static_cast<std::size_t>(heap_pixel[0])] = 0;
                std::size_t node = 0;
                for (;;) {
                    const std::size_t left = 2 * node + 1;
                    const std::size_t right = left + 1;
                    std::size_t smallest = node;
                    if (left < heap_size &&
                        heap_value[left] < heap_value[smallest]) {
                        smallest = left;
                    }
                    if (right < heap_size &&
                        heap_value[right] < heap_value[smallest]) {
                        smallest = right;
                    }
                    if (smallest == node) {
                        break;
                    }
                    std::swap(heap_value[node], heap_value[smallest]);
                    std::swap(heap_pixel[node], heap_pixel[smallest]);
                    heap_position[static_cast<std::size_t>(
                        heap_pixel[node])] = static_cast<std::int32_t>(node);
                    heap_position[static_cast<std::size_t>(
                        heap_pixel[smallest])] =
                            static_cast<std::int32_t>(smallest);
                    node = smallest;
                }
            }

            const std::int32_t label = tentative_label[pixel];
            accepted[pixel] = 1;
            distance[pixel] = value;
            owner[pixel] = label;
            if (full_output) {
                gradient_x[pixel] = tentative_gradient_x[pixel];
                gradient_y[pixel] = tentative_gradient_y[pixel];
                parent_first[pixel] = tentative_parent_first[pixel];
                parent_second[pixel] = tentative_parent_second[pixel];
                parent_fraction[pixel] = tentative_parent_fraction[pixel];
                acceptance_order[acceptance_size] =
                    static_cast<std::int32_t>(pixel);
            }
            if (source_output) {
                source_gradient_x[pixel] =
                    tentative_source_gradient_x[pixel];
                source_gradient_y[pixel] =
                    tentative_source_gradient_y[pixel];
            }
            ++acceptance_size;

            const std::size_t begin =
                static_cast<std::size_t>(inverse_offset[pixel]);
            const std::size_t end =
                static_cast<std::size_t>(inverse_offset[pixel + 1]);
            for (std::size_t incidence = begin;
                 incidence < end; ++incidence) {
                const std::size_t receiver =
                    static_cast<std::size_t>(inverse_receiver[incidence]);
                if (accepted[receiver] != 0) {
                    continue;
                }
                const double a = mxx[receiver];
                const double b = mxy[receiver];
                const double c = myy[receiver];
                double best_value = tentative[receiver];
                std::int32_t best_label = tentative_label[receiver];
                double best_gradient_x = tentative_gradient_x[receiver];
                double best_gradient_y = tentative_gradient_y[receiver];
                double best_source_gradient_x = 0.0;
                double best_source_gradient_y = 0.0;
                if (source_output) {
                    best_source_gradient_x =
                        tentative_source_gradient_x[receiver];
                    best_source_gradient_y =
                        tentative_source_gradient_y[receiver];
                }
                std::int32_t best_parent_first =
                    tentative_parent_first[receiver];
                std::int32_t best_parent_second =
                    tentative_parent_second[receiver];
                double best_parent_fraction =
                    tentative_parent_fraction[receiver];

                const std::size_t cardinal_base = receiver * 4;
                for (std::size_t cardinal = 0;
                     cardinal < 4; ++cardinal) {
                    const int ux = cardinal_x[cardinal];
                    const int uy = cardinal_y[cardinal];
                    const double cardinal_cost =
                        cardinal_costs[cardinal_base + cardinal];
                    if (!std::isfinite(cardinal_cost)) {
                        continue;
                    }
                    const std::size_t neighbour =
                        static_cast<std::size_t>(
                            static_cast<std::ptrdiff_t>(receiver) +
                            static_cast<std::ptrdiff_t>(uy) *
                                static_cast<std::ptrdiff_t>(width) + ux);
                    if (accepted[neighbour] == 0) {
                        continue;
                    }
                    const double candidate =
                        distance[neighbour] +
                        cardinal_cost;
                    if (candidate < best_value) {
                        best_value = candidate;
                        best_label = owner[neighbour];
                        const double local_length =
                            metric_norm(ux, uy, a, b, c);
                        best_gradient_x =
                            -(a * ux + b * uy) / local_length;
                        best_gradient_y =
                            -(b * ux + c * uy) / local_length;
                        if (source_output) {
                            best_source_gradient_x =
                                source_gradient_x[neighbour];
                            best_source_gradient_y =
                                source_gradient_y[neighbour];
                        }
                        best_parent_first =
                            static_cast<std::int32_t>(neighbour);
                        best_parent_second = -1;
                        best_parent_fraction = 0.0;
                    }
                }

                const std::size_t direction_base = receiver * 6;
                for (std::size_t direction = 0;
                     direction < 6; ++direction) {
                    const std::size_t next =
                        direction == 5 ? 0 : direction + 1;
                    const std::size_t first_vector =
                        (direction_base + direction) * 2;
                    const std::size_t second_vector =
                        (direction_base + next) * 2;
                    const int ux = directions[first_vector];
                    const int uy = directions[first_vector + 1];
                    const int vx = directions[second_vector];
                    const int vy = directions[second_vector + 1];
                    const bool first_valid = direction_valid[
                        direction_base + direction] != 0;
                    const bool second_valid = direction_valid[
                        direction_base + next] != 0;
                    const std::size_t first_pixel = first_valid
                        ? static_cast<std::size_t>(
                            static_cast<std::ptrdiff_t>(receiver) +
                            static_cast<std::ptrdiff_t>(uy) *
                                static_cast<std::ptrdiff_t>(width) + ux)
                        : pixels;
                    const std::size_t second_pixel = second_valid
                        ? static_cast<std::size_t>(
                            static_cast<std::ptrdiff_t>(receiver) +
                            static_cast<std::ptrdiff_t>(vy) *
                                static_cast<std::ptrdiff_t>(width) + vx)
                        : pixels;

                    if (first_valid && accepted[first_pixel] != 0) {
                        const double candidate =
                            distance[first_pixel] +
                            direction_costs[direction_base + direction];
                        if (candidate < best_value) {
                            best_value = candidate;
                            best_label = owner[first_pixel];
                            const double local_length =
                                metric_norm(ux, uy, a, b, c);
                            best_gradient_x =
                                -(a * ux + b * uy) / local_length;
                            best_gradient_y =
                                -(b * ux + c * uy) / local_length;
                            if (source_output) {
                                best_source_gradient_x =
                                    source_gradient_x[first_pixel];
                                best_source_gradient_y =
                                    source_gradient_y[first_pixel];
                            }
                            best_parent_first =
                                static_cast<std::int32_t>(first_pixel);
                            best_parent_second = -1;
                            best_parent_fraction = 0.0;
                        }
                    }
                    if (second_valid && accepted[second_pixel] != 0) {
                        const double candidate =
                            distance[second_pixel] +
                            direction_costs[direction_base + next];
                        if (candidate < best_value) {
                            best_value = candidate;
                            best_label = owner[second_pixel];
                            const double local_length =
                                metric_norm(vx, vy, a, b, c);
                            best_gradient_x =
                                -(a * vx + b * vy) / local_length;
                            best_gradient_y =
                                -(b * vx + c * vy) / local_length;
                            if (source_output) {
                                best_source_gradient_x =
                                    source_gradient_x[second_pixel];
                                best_source_gradient_y =
                                    source_gradient_y[second_pixel];
                            }
                            best_parent_first =
                                static_cast<std::int32_t>(second_pixel);
                            best_parent_second = -1;
                            best_parent_fraction = 0.0;
                        }
                    }
                    if (first_valid && second_valid &&
                        accepted[first_pixel] != 0 &&
                        accepted[second_pixel] != 0 &&
                        owner[first_pixel] == owner[second_pixel]) {
                        double fraction = 0.0;
                        const double candidate = simplex_candidate(
                            distance[first_pixel],
                            distance[second_pixel],
                            ux,
                            uy,
                            vx,
                            vy,
                            a,
                            b,
                            c,
                            &fraction);
                        if (candidate < best_value) {
                            best_value = candidate;
                            best_label = owner[first_pixel];
                            const double foot_x =
                                ux + fraction * (vx - ux);
                            const double foot_y =
                                uy + fraction * (vy - uy);
                            const double local_length =
                                metric_norm(foot_x, foot_y, a, b, c);
                            best_gradient_x = -(
                                a * foot_x + b * foot_y) / local_length;
                            best_gradient_y = -(
                                b * foot_x + c * foot_y) / local_length;
                            if (source_output) {
                                best_source_gradient_x =
                                    (1.0 - fraction) *
                                        source_gradient_x[first_pixel] +
                                    fraction *
                                        source_gradient_x[second_pixel];
                                best_source_gradient_y =
                                    (1.0 - fraction) *
                                        source_gradient_y[first_pixel] +
                                    fraction *
                                        source_gradient_y[second_pixel];
                            }
                            best_parent_first =
                                static_cast<std::int32_t>(first_pixel);
                            best_parent_second =
                                static_cast<std::int32_t>(second_pixel);
                            best_parent_fraction = fraction;
                        }
                    }
                }

                if (best_value + 1e-12 >= tentative[receiver]) {
                    continue;
                }
                tentative[receiver] = best_value;
                tentative_label[receiver] = best_label;
                tentative_gradient_x[receiver] = best_gradient_x;
                tentative_gradient_y[receiver] = best_gradient_y;
                if (source_output) {
                    tentative_source_gradient_x[receiver] =
                        best_source_gradient_x;
                    tentative_source_gradient_y[receiver] =
                        best_source_gradient_y;
                }
                tentative_parent_first[receiver] = best_parent_first;
                tentative_parent_second[receiver] = best_parent_second;
                tentative_parent_fraction[receiver] = best_parent_fraction;
                std::int32_t child_position = heap_position[receiver];
                if (child_position < 0) {
                    child_position = static_cast<std::int32_t>(heap_size);
                    heap_pixel[heap_size] =
                        static_cast<std::int32_t>(receiver);
                    heap_position[receiver] = child_position;
                    ++heap_size;
                }
                const std::size_t child =
                    static_cast<std::size_t>(child_position);
                heap_value[child] = best_value;
                ++pushes;
                maximum_heap = std::max(maximum_heap, heap_size);
                bubble_up(child);
            }
        }

        if (full_output) {
            *accepted_count = acceptance_size;
        }
        *push_count = pushes;
        *maximum_heap_size = maximum_heap;
        return BFFT_OK;
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
}

bfft_status bfft_vision_fast_march_labels(
    std::size_t height,
    std::size_t width,
    std::size_t seed_count,
    const std::int32_t* seed_pixel,
    const double* seed_value,
    const std::int32_t* seed_label,
    const double* seed_gradient_x,
    const double* seed_gradient_y,
    const std::int32_t* directions,
    const double* direction_costs,
    const std::uint8_t* direction_valid,
    const double* cardinal_costs,
    const std::int64_t* inverse_offset,
    std::size_t inverse_count,
    const std::int32_t* inverse_receiver,
    const double* mxx,
    const double* mxy,
    const double* myy,
    std::int32_t* owner,
    double* distance,
    std::size_t* push_count,
    std::size_t* maximum_heap_size) {
    std::size_t pixels = 0;
    if (height == 0 || width == 0 ||
        !checked_product(height, width, &pixels) ||
        seed_pixel == nullptr || seed_value == nullptr ||
        seed_label == nullptr || seed_gradient_x == nullptr ||
        seed_gradient_y == nullptr || directions == nullptr ||
        direction_costs == nullptr || direction_valid == nullptr ||
        cardinal_costs == nullptr || inverse_offset == nullptr ||
        (inverse_count != 0 && inverse_receiver == nullptr) ||
        mxx == nullptr || mxy == nullptr || myy == nullptr ||
        owner == nullptr || distance == nullptr ||
        push_count == nullptr || maximum_heap_size == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    if (inverse_offset[0] != 0 ||
        inverse_offset[pixels] < 0 ||
        static_cast<std::size_t>(inverse_offset[pixels]) != inverse_count) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    for (std::size_t seed = 0; seed < seed_count; ++seed) {
        if (seed_pixel[seed] < 0 ||
            static_cast<std::size_t>(seed_pixel[seed]) >= pixels ||
            seed_label[seed] < 0) {
            return BFFT_ERROR_INVALID_ARGUMENT;
        }
    }

    constexpr double infinity = 1e300;
    const auto metric_norm = [](
        double dx, double dy, double a, double b, double c) {
        return std::sqrt(std::max(
            a * dx * dx + 2.0 * b * dx * dy + c * dy * dy,
            1e-30));
    };
    const auto simplex_candidate = [&metric_norm](
        double first_value,
        double second_value,
        double first_x,
        double first_y,
        double second_x,
        double second_y,
        double a,
        double b,
        double c) {
        const double delta_value = second_value - first_value;
        const double delta_x = second_x - first_x;
        const double delta_y = second_y - first_y;
        const double quadratic_a =
            a * delta_x * delta_x +
            2.0 * b * delta_x * delta_y +
            c * delta_y * delta_y;
        const double quadratic_b =
            delta_x * (a * first_x + b * first_y) +
            delta_y * (b * first_x + c * first_y);
        const double quadratic_c =
            a * first_x * first_x +
            2.0 * b * first_x * first_y +
            c * first_y * first_y;
        const double first_length =
            std::sqrt(std::max(quadratic_c, 1e-30));
        const double derivative_first =
            delta_value + quadratic_b / first_length;
        if (derivative_first >= 0.0) {
            return first_value + first_length;
        }
        const double second_quadratic =
            quadratic_a + 2.0 * quadratic_b + quadratic_c;
        const double second_length =
            std::sqrt(std::max(second_quadratic, 1e-30));
        const double derivative_second =
            delta_value +
            (quadratic_a + quadratic_b) / second_length;
        if (derivative_second <= 0.0) {
            return second_value + second_length;
        }
        const double area = std::max(
            quadratic_a * quadratic_c - quadratic_b * quadratic_b,
            0.0);
        const double causal = std::max(
            quadratic_a - delta_value * delta_value,
            1e-30);
        double fraction = (
            -quadratic_b -
            delta_value * std::sqrt(area / causal)
        ) / std::max(quadratic_a, 1e-30);
        fraction = std::min(std::max(fraction, 0.0), 1.0);
        const double rx = first_x + fraction * delta_x;
        const double ry = first_y + fraction * delta_y;
        return (
            first_value +
            fraction * delta_value +
            metric_norm(rx, ry, a, b, c));
    };

    try {
        std::vector<double> tentative(pixels, infinity);
        std::vector<std::int32_t> tentative_label(pixels, -1);
        std::vector<std::uint8_t> accepted(pixels, 0);
        std::vector<double> heap_value(pixels);
        std::vector<std::int32_t> heap_pixel(pixels);
        std::vector<std::int32_t> heap_position(pixels, -1);
        std::fill(distance, distance + pixels, infinity);
        std::fill(owner, owner + pixels, std::int32_t{-1});

        std::size_t heap_size = 0;
        std::size_t pushes = 0;
        std::size_t maximum_heap = 0;
        const auto bubble_up = [&] (std::size_t child) {
            while (child > 0) {
                const std::size_t parent = (child - 1) / 2;
                if (heap_value[parent] <= heap_value[child]) {
                    break;
                }
                std::swap(heap_value[parent], heap_value[child]);
                std::swap(heap_pixel[parent], heap_pixel[child]);
                heap_position[static_cast<std::size_t>(
                    heap_pixel[parent])] = static_cast<std::int32_t>(parent);
                heap_position[static_cast<std::size_t>(
                    heap_pixel[child])] = static_cast<std::int32_t>(child);
                child = parent;
            }
        };

        for (std::size_t seed = 0; seed < seed_count; ++seed) {
            const std::size_t pixel =
                static_cast<std::size_t>(seed_pixel[seed]);
            const double value = seed_value[seed];
            if (value >= tentative[pixel]) {
                continue;
            }
            tentative[pixel] = value;
            tentative_label[pixel] = seed_label[seed];
            std::int32_t child_position = heap_position[pixel];
            if (child_position < 0) {
                child_position = static_cast<std::int32_t>(heap_size);
                heap_pixel[heap_size] = static_cast<std::int32_t>(pixel);
                heap_position[pixel] = child_position;
                ++heap_size;
            }
            const std::size_t child =
                static_cast<std::size_t>(child_position);
            heap_value[child] = value;
            ++pushes;
            maximum_heap = std::max(maximum_heap, heap_size);
            bubble_up(child);
        }

        static constexpr int cardinal_x[4] = {1, -1, 0, 0};
        static constexpr int cardinal_y[4] = {0, 0, 1, -1};
        while (heap_size > 0) {
            const double value = heap_value[0];
            const std::size_t pixel =
                static_cast<std::size_t>(heap_pixel[0]);
            --heap_size;
            heap_position[pixel] = -2;
            if (heap_size > 0) {
                heap_value[0] = heap_value[heap_size];
                heap_pixel[0] = heap_pixel[heap_size];
                heap_position[static_cast<std::size_t>(heap_pixel[0])] = 0;
                std::size_t node = 0;
                for (;;) {
                    const std::size_t left = 2 * node + 1;
                    const std::size_t right = left + 1;
                    std::size_t smallest = node;
                    if (left < heap_size &&
                        heap_value[left] < heap_value[smallest]) {
                        smallest = left;
                    }
                    if (right < heap_size &&
                        heap_value[right] < heap_value[smallest]) {
                        smallest = right;
                    }
                    if (smallest == node) {
                        break;
                    }
                    std::swap(heap_value[node], heap_value[smallest]);
                    std::swap(heap_pixel[node], heap_pixel[smallest]);
                    heap_position[static_cast<std::size_t>(
                        heap_pixel[node])] = static_cast<std::int32_t>(node);
                    heap_position[static_cast<std::size_t>(
                        heap_pixel[smallest])] =
                            static_cast<std::int32_t>(smallest);
                    node = smallest;
                }
            }

            accepted[pixel] = 1;
            distance[pixel] = value;
            owner[pixel] = tentative_label[pixel];
            const std::size_t begin =
                static_cast<std::size_t>(inverse_offset[pixel]);
            const std::size_t end =
                static_cast<std::size_t>(inverse_offset[pixel + 1]);
            for (std::size_t incidence = begin;
                 incidence < end; ++incidence) {
                const std::size_t receiver =
                    static_cast<std::size_t>(inverse_receiver[incidence]);
                if (accepted[receiver] != 0) {
                    continue;
                }
                const double a = mxx[receiver];
                const double b = mxy[receiver];
                const double c = myy[receiver];
                double best_value = tentative[receiver];
                std::int32_t best_label = tentative_label[receiver];

                const std::size_t cardinal_base = receiver * 4;
                for (std::size_t cardinal = 0;
                     cardinal < 4; ++cardinal) {
                    const int ux = cardinal_x[cardinal];
                    const int uy = cardinal_y[cardinal];
                    const double cardinal_cost =
                        cardinal_costs[cardinal_base + cardinal];
                    if (!std::isfinite(cardinal_cost)) {
                        continue;
                    }
                    const std::size_t neighbour =
                        static_cast<std::size_t>(
                            static_cast<std::ptrdiff_t>(receiver) +
                            static_cast<std::ptrdiff_t>(uy) *
                                static_cast<std::ptrdiff_t>(width) + ux);
                    if (accepted[neighbour] == 0) {
                        continue;
                    }
                    const double candidate =
                        distance[neighbour] +
                        cardinal_cost;
                    if (candidate < best_value) {
                        best_value = candidate;
                        best_label = owner[neighbour];
                    }
                }

                const std::size_t direction_base = receiver * 6;
                for (std::size_t direction = 0;
                     direction < 6; ++direction) {
                    const std::size_t next =
                        direction == 5 ? 0 : direction + 1;
                    const std::size_t first_vector =
                        (direction_base + direction) * 2;
                    const std::size_t second_vector =
                        (direction_base + next) * 2;
                    const int ux = directions[first_vector];
                    const int uy = directions[first_vector + 1];
                    const int vx = directions[second_vector];
                    const int vy = directions[second_vector + 1];
                    const bool first_valid = direction_valid[
                        direction_base + direction] != 0;
                    const bool second_valid = direction_valid[
                        direction_base + next] != 0;
                    const std::size_t first_pixel = first_valid
                        ? static_cast<std::size_t>(
                            static_cast<std::ptrdiff_t>(receiver) +
                            static_cast<std::ptrdiff_t>(uy) *
                                static_cast<std::ptrdiff_t>(width) + ux)
                        : pixels;
                    const std::size_t second_pixel = second_valid
                        ? static_cast<std::size_t>(
                            static_cast<std::ptrdiff_t>(receiver) +
                            static_cast<std::ptrdiff_t>(vy) *
                                static_cast<std::ptrdiff_t>(width) + vx)
                        : pixels;

                    if (first_valid && accepted[first_pixel] != 0) {
                        const double candidate =
                            distance[first_pixel] +
                            direction_costs[direction_base + direction];
                        if (candidate < best_value) {
                            best_value = candidate;
                            best_label = owner[first_pixel];
                        }
                    }
                    if (second_valid && accepted[second_pixel] != 0) {
                        const double candidate =
                            distance[second_pixel] +
                            direction_costs[direction_base + next];
                        if (candidate < best_value) {
                            best_value = candidate;
                            best_label = owner[second_pixel];
                        }
                    }
                    if (first_valid && second_valid &&
                        accepted[first_pixel] != 0 &&
                        accepted[second_pixel] != 0 &&
                        owner[first_pixel] == owner[second_pixel]) {
                        const double candidate = simplex_candidate(
                            distance[first_pixel],
                            distance[second_pixel],
                            ux,
                            uy,
                            vx,
                            vy,
                            a,
                            b,
                            c);
                        if (candidate < best_value) {
                            best_value = candidate;
                            best_label = owner[first_pixel];
                        }
                    }
                }

                if (best_value + 1e-12 >= tentative[receiver]) {
                    continue;
                }
                tentative[receiver] = best_value;
                tentative_label[receiver] = best_label;
                std::int32_t child_position = heap_position[receiver];
                if (child_position < 0) {
                    child_position = static_cast<std::int32_t>(heap_size);
                    heap_pixel[heap_size] =
                        static_cast<std::int32_t>(receiver);
                    heap_position[receiver] = child_position;
                    ++heap_size;
                }
                const std::size_t child =
                    static_cast<std::size_t>(child_position);
                heap_value[child] = best_value;
                ++pushes;
                maximum_heap = std::max(maximum_heap, heap_size);
                bubble_up(child);
            }
        }
        *push_count = pushes;
        *maximum_heap_size = maximum_heap;
        return BFFT_OK;
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
}

bfft_status bfft_vision_metric_edge_costs_f32(
    std::size_t height,
    std::size_t width,
    const float* precision_xx,
    const float* precision_xy,
    const float* precision_yy,
    const float* boundary_xx,
    const float* boundary_xy,
    const float* boundary_yy,
    double precision_gain,
    double boundary_gain,
    float* direction_costs) {
    std::size_t pixels = 0;
    const bool has_boundary = boundary_gain > 0.0;
    if (height == 0 || width == 0 ||
        !checked_product(height, width, &pixels) ||
        precision_xx == nullptr || precision_xy == nullptr ||
        precision_yy == nullptr || direction_costs == nullptr ||
        !std::isfinite(precision_gain) || precision_gain < 0.0 ||
        !std::isfinite(boundary_gain) || boundary_gain < 0.0 ||
        (has_boundary && (
            boundary_xx == nullptr || boundary_xy == nullptr ||
            boundary_yy == nullptr))) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }

    static constexpr int dx[8] = {0, 0, -1, 1, -1, 1, -1, 1};
    static constexpr int dy[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    const float infinity = std::numeric_limits<float>::infinity();
    std::fill(
        direction_costs, direction_costs + 8 * pixels, infinity);
    const auto metric_component = [
        precision_gain,
        boundary_gain,
        has_boundary
    ](
        const float* precision,
        const float* boundary,
        std::size_t pixel,
        double identity) {
        double value =
            identity +
            precision_gain * static_cast<double>(precision[pixel]);
        if (has_boundary) {
            value += boundary_gain *
                static_cast<double>(boundary[pixel]);
        }
        return value;
    };
    const auto metric_at = [
        &metric_component,
        precision_xx,
        precision_xy,
        precision_yy
    ](
        std::size_t pixel,
        const float* boundary_xx_values,
        const float* boundary_xy_values,
        const float* boundary_yy_values,
        double* xx,
        double* xy,
        double* yy) {
        const double raw_xx = metric_component(
            precision_xx, boundary_xx_values, pixel, 1.0);
        const double raw_xy = metric_component(
            precision_xy, boundary_xy_values, pixel, 0.0);
        const double raw_yy = metric_component(
            precision_yy, boundary_yy_values, pixel, 1.0);
        const double excess_xx = std::max(raw_xx - 1.0, 0.0);
        const double excess_yy = std::max(raw_yy - 1.0, 0.0);
        const double cross_limit = std::sqrt(excess_xx * excess_yy);
        *xx = 1.0 + excess_xx;
        *xy = std::max(-cross_limit, std::min(raw_xy, cross_limit));
        *yy = 1.0 + excess_yy;
    };
    for (std::size_t direction = 0; direction < 8; ++direction) {
        const int step_x = dx[direction];
        const int step_y = dy[direction];
        const std::size_t y_begin =
            step_y < 0 ? static_cast<std::size_t>(-step_y) : 0;
        const std::size_t y_end =
            step_y > 0 ? height - static_cast<std::size_t>(step_y) : height;
        const std::size_t x_begin =
            step_x < 0 ? static_cast<std::size_t>(-step_x) : 0;
        const std::size_t x_end =
            step_x > 0 ? width - static_cast<std::size_t>(step_x) : width;
        float* output = direction_costs + direction * pixels;
        for (std::size_t y = y_begin; y < y_end; ++y) {
            for (std::size_t x = x_begin; x < x_end; ++x) {
                const std::size_t source = y * width + x;
                const std::size_t receiver =
                    static_cast<std::size_t>(
                        static_cast<std::int64_t>(y) + step_y) * width +
                    static_cast<std::size_t>(
                        static_cast<std::int64_t>(x) + step_x);
                double source_xx = 0.0;
                double source_xy = 0.0;
                double source_yy = 0.0;
                double receiver_xx = 0.0;
                double receiver_xy = 0.0;
                double receiver_yy = 0.0;
                metric_at(
                    source,
                    boundary_xx,
                    boundary_xy,
                    boundary_yy,
                    &source_xx,
                    &source_xy,
                    &source_yy);
                metric_at(
                    receiver,
                    boundary_xx,
                    boundary_xy,
                    boundary_yy,
                    &receiver_xx,
                    &receiver_xy,
                    &receiver_yy);
                const double a = 0.5 * (source_xx + receiver_xx);
                const double b = 0.5 * (source_xy + receiver_xy);
                const double c = 0.5 * (source_yy + receiver_yy);
                const double quadratic =
                    step_x * step_x * a +
                    2.0 * step_x * step_y * b +
                    step_y * step_y * c;
                output[source] = static_cast<float>(
                    std::sqrt(std::max(quadratic, 1e-8)));
            }
        }
    }
    return BFFT_OK;
}

bfft_status bfft_vision_bucket_first_label(
    std::size_t height,
    std::size_t width,
    std::size_t seed_count,
    const std::int64_t* seed_pixel,
    const double* reach,
    const float* direction_costs,
    double delta,
    std::size_t span,
    double shift,
    std::int32_t* owner,
    double* distance,
    std::int32_t* parent,
    std::size_t* push_count) {
    std::size_t pixels = 0;
    if (height == 0 || width == 0 ||
        !checked_product(height, width, &pixels) ||
        seed_pixel == nullptr || reach == nullptr ||
        direction_costs == nullptr || !(delta > 0.0) ||
        span == 0 || owner == nullptr || distance == nullptr ||
        parent == nullptr || push_count == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    for (std::size_t site = 0; site < seed_count; ++site) {
        if (seed_pixel[site] < 0 ||
            static_cast<std::size_t>(seed_pixel[site]) >= pixels) {
            return BFFT_ERROR_INVALID_ARGUMENT;
        }
    }

    constexpr double infinity = 1e300;
    constexpr double tolerance = 1e-12;
    static constexpr int dx[8] = {0, 0, -1, 1, -1, 1, -1, 1};
    static constexpr int dy[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    try {
        const std::size_t bucket_count = span + 2;
        std::vector<std::int32_t> head(bucket_count, -1);
        std::vector<double> key;
        std::vector<std::int32_t> entry_pixel;
        std::vector<std::int32_t> next;
        const std::size_t reserve =
            std::min(
                pixels * 2 + 256,
                std::numeric_limits<std::size_t>::max() / 4);
        key.reserve(reserve);
        entry_pixel.reserve(reserve);
        next.reserve(reserve);
        std::fill(distance, distance + pixels, infinity);
        std::fill(owner, owner + pixels, std::int32_t{-1});
        std::fill(parent, parent + pixels, std::int32_t{-1});

        std::size_t alive = 0;
        const auto insert = [&](
            double value, std::size_t pixel) {
            const std::int64_t slot = static_cast<std::int64_t>(
                (value + shift) / delta);
            const std::size_t bucket = static_cast<std::size_t>(
                slot % static_cast<std::int64_t>(bucket_count));
            const std::int32_t index =
                static_cast<std::int32_t>(key.size());
            key.push_back(value);
            entry_pixel.push_back(static_cast<std::int32_t>(pixel));
            next.push_back(head[bucket]);
            head[bucket] = index;
            ++alive;
        };

        for (std::size_t site = 0; site < seed_count; ++site) {
            const std::size_t pixel =
                static_cast<std::size_t>(seed_pixel[site]);
            const double value = -reach[site];
            if (value < distance[pixel]) {
                distance[pixel] = value;
                owner[pixel] = static_cast<std::int32_t>(site);
                parent[pixel] = -1;
                insert(value, pixel);
            }
        }

        std::size_t current = 0;
        std::size_t guard = 0;
        const std::size_t limit =
            bucket_count * (pixels + 16);
        while (alive > 0 && guard < limit) {
            const std::size_t bucket = current % bucket_count;
            std::int32_t entry = head[bucket];
            if (entry < 0) {
                ++current;
                ++guard;
                continue;
            }
            head[bucket] = -1;
            while (entry >= 0) {
                const std::size_t index =
                    static_cast<std::size_t>(entry);
                const double value = key[index];
                const std::size_t pixel = static_cast<std::size_t>(
                    entry_pixel[index]);
                entry = next[index];
                --alive;
                if (value > distance[pixel] + tolerance) {
                    continue;
                }
                const std::int32_t site = owner[pixel];

                const std::size_t y = pixel / width;
                const std::size_t x = pixel - y * width;
                for (std::size_t direction = 0;
                     direction < 8; ++direction) {
                    const std::int64_t nx =
                        static_cast<std::int64_t>(x) + dx[direction];
                    const std::int64_t ny =
                        static_cast<std::int64_t>(y) + dy[direction];
                    if (
                        nx < 0 || nx >= static_cast<std::int64_t>(width) ||
                        ny < 0 || ny >= static_cast<std::int64_t>(height)
                    ) {
                        continue;
                    }
                    const std::size_t receiver =
                        static_cast<std::size_t>(ny) * width +
                        static_cast<std::size_t>(nx);
                    const double candidate =
                        value + static_cast<double>(
                            direction_costs[
                                direction * pixels + pixel]);
                    if (candidate + tolerance >= distance[receiver]) {
                        continue;
                    }
                    distance[receiver] = candidate;
                    owner[receiver] = site;
                    parent[receiver] =
                        static_cast<std::int32_t>(pixel);
                    insert(candidate, receiver);
                }
            }
            ++current;
            ++guard;
        }
        *push_count = key.size();
        return BFFT_OK;
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
}

bfft_status bfft_vision_bucket_two_labels(
    std::size_t height,
    std::size_t width,
    std::size_t seed_count,
    const std::int64_t* seed_pixel,
    const double* reach,
    const float* direction_costs,
    double delta,
    std::size_t span,
    double shift,
    std::int32_t* owner,
    std::int32_t* runner,
    double* distance,
    double* second_distance,
    std::int32_t* parent_first,
    std::size_t* push_count) {
    std::size_t pixels = 0;
    if (height == 0 || width == 0 || seed_count == 0 ||
        seed_count > static_cast<std::size_t>(
            std::numeric_limits<std::int32_t>::max()) ||
        !checked_product(height, width, &pixels) ||
        pixels > static_cast<std::size_t>(
            std::numeric_limits<std::int32_t>::max()) ||
        seed_pixel == nullptr || reach == nullptr ||
        direction_costs == nullptr || !(delta > 0.0) || span == 0 ||
        owner == nullptr || runner == nullptr || distance == nullptr ||
        second_distance == nullptr || parent_first == nullptr ||
        push_count == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    for (std::size_t site = 0; site < seed_count; ++site) {
        if (seed_pixel[site] < 0 ||
            static_cast<std::size_t>(seed_pixel[site]) >= pixels ||
            !std::isfinite(reach[site])) {
            return BFFT_ERROR_INVALID_ARGUMENT;
        }
    }

    constexpr double infinity = 1e300;
    constexpr double tolerance = 1e-12;
    static constexpr int dx[8] = {0, 0, -1, 1, -1, 1, -1, 1};
    static constexpr int dy[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    try {
        const std::size_t bucket_count = span + 2;
        std::vector<std::int32_t> head(bucket_count, -1);
        std::vector<double> key;
        std::vector<std::int32_t> entry_pixel;
        std::vector<std::int32_t> entry_site;
        std::vector<std::int32_t> next;
        std::vector<std::int32_t> parent_second(pixels, -1);
        const std::size_t reserve = std::min(
            pixels * 4 + 256,
            static_cast<std::size_t>(
                std::numeric_limits<std::int32_t>::max()));
        key.reserve(reserve);
        entry_pixel.reserve(reserve);
        entry_site.reserve(reserve);
        next.reserve(reserve);
        std::fill(distance, distance + pixels, infinity);
        std::fill(second_distance, second_distance + pixels, infinity);
        std::fill(owner, owner + pixels, std::int32_t{-1});
        std::fill(runner, runner + pixels, std::int32_t{-1});
        std::fill(parent_first, parent_first + pixels, std::int32_t{-1});

        std::size_t alive = 0;
        const auto insert = [&] (
            double value, std::size_t pixel, std::int32_t site) -> bool {
            if (key.size() >= static_cast<std::size_t>(
                    std::numeric_limits<std::int32_t>::max())) {
                return false;
            }
            const std::int64_t slot = static_cast<std::int64_t>(
                (value + shift) / delta);
            const std::size_t bucket = static_cast<std::size_t>(
                slot % static_cast<std::int64_t>(bucket_count));
            const std::int32_t index =
                static_cast<std::int32_t>(key.size());
            key.push_back(value);
            entry_pixel.push_back(static_cast<std::int32_t>(pixel));
            entry_site.push_back(site);
            next.push_back(head[bucket]);
            head[bucket] = index;
            ++alive;
            return true;
        };

        for (std::size_t site_index = 0;
             site_index < seed_count; ++site_index) {
            const std::size_t pixel =
                static_cast<std::size_t>(seed_pixel[site_index]);
            const std::int32_t site =
                static_cast<std::int32_t>(site_index);
            const double value = -reach[site_index];
            if (value < distance[pixel]) {
                second_distance[pixel] = distance[pixel];
                runner[pixel] = owner[pixel];
                parent_second[pixel] = parent_first[pixel];
                distance[pixel] = value;
                owner[pixel] = site;
                parent_first[pixel] = -1;
            } else if (site != owner[pixel] &&
                       value < second_distance[pixel]) {
                second_distance[pixel] = value;
                runner[pixel] = site;
                parent_second[pixel] = -1;
            }
            if (!insert(value, pixel, site)) {
                return BFFT_ERROR_ALLOCATION;
            }
        }

        std::size_t current = 0;
        std::size_t guard = 0;
        const std::size_t limit = bucket_count * (pixels + 16);
        while (alive > 0 && guard < limit) {
            const std::size_t bucket = current % bucket_count;
            std::int32_t entry = head[bucket];
            if (entry < 0) {
                ++current;
                ++guard;
                continue;
            }
            head[bucket] = -1;
            while (entry >= 0) {
                const std::size_t index =
                    static_cast<std::size_t>(entry);
                const double value = key[index];
                const std::size_t pixel = static_cast<std::size_t>(
                    entry_pixel[index]);
                const std::int32_t site = entry_site[index];
                entry = next[index];
                --alive;
                if (!(
                    (owner[pixel] == site &&
                     value <= distance[pixel] + tolerance) ||
                    (runner[pixel] == site &&
                     value <= second_distance[pixel] + tolerance))) {
                    continue;
                }

                const std::size_t y = pixel / width;
                const std::size_t x = pixel - y * width;
                for (std::size_t direction = 0;
                     direction < 8; ++direction) {
                    const std::int64_t nx =
                        static_cast<std::int64_t>(x) + dx[direction];
                    const std::int64_t ny =
                        static_cast<std::int64_t>(y) + dy[direction];
                    if (nx < 0 || nx >= static_cast<std::int64_t>(width) ||
                        ny < 0 || ny >= static_cast<std::int64_t>(height)) {
                        continue;
                    }
                    const std::size_t receiver =
                        static_cast<std::size_t>(ny) * width +
                        static_cast<std::size_t>(nx);
                    const double candidate = value + static_cast<double>(
                        direction_costs[direction * pixels + pixel]);
                    bool touched = false;
                    if (owner[receiver] == site) {
                        if (candidate + tolerance < distance[receiver]) {
                            distance[receiver] = candidate;
                            parent_first[receiver] =
                                static_cast<std::int32_t>(pixel);
                            touched = true;
                        }
                    } else if (runner[receiver] == site) {
                        if (candidate + tolerance < second_distance[receiver]) {
                            second_distance[receiver] = candidate;
                            parent_second[receiver] =
                                static_cast<std::int32_t>(pixel);
                            if (second_distance[receiver] < distance[receiver]) {
                                std::swap(
                                    distance[receiver],
                                    second_distance[receiver]);
                                std::swap(owner[receiver], runner[receiver]);
                                std::swap(
                                    parent_first[receiver],
                                    parent_second[receiver]);
                            }
                            touched = true;
                        }
                    } else if (candidate + tolerance < distance[receiver]) {
                        second_distance[receiver] = distance[receiver];
                        runner[receiver] = owner[receiver];
                        parent_second[receiver] = parent_first[receiver];
                        distance[receiver] = candidate;
                        owner[receiver] = site;
                        parent_first[receiver] =
                            static_cast<std::int32_t>(pixel);
                        touched = true;
                    } else if (
                        candidate + tolerance < second_distance[receiver]) {
                        second_distance[receiver] = candidate;
                        runner[receiver] = site;
                        parent_second[receiver] =
                            static_cast<std::int32_t>(pixel);
                        touched = true;
                    }
                    if (touched && !insert(candidate, receiver, site)) {
                        return BFFT_ERROR_ALLOCATION;
                    }
                }
            }
            ++current;
            ++guard;
        }
        *push_count = key.size();
        return BFFT_OK;
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
}

bfft_status bfft_vision_hard_affine_fit(
    std::size_t height,
    std::size_t width,
    std::size_t cell_count,
    const std::int32_t* labels,
    const double* target,
    double* basis,
    double* count,
    double* radius,
    double* centroid,
    double* reconstruction) {
    std::size_t pixels = 0;
    if (height == 0 || width == 0 || cell_count == 0 ||
        !checked_product(height, width, &pixels) ||
        labels == nullptr || target == nullptr || basis == nullptr ||
        count == nullptr || radius == nullptr || centroid == nullptr ||
        reconstruction == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    try {
        std::vector<double> centroid_sum(cell_count * 2, 0.0);
        std::vector<double> second_moment(cell_count * 3, 0.0);
        std::vector<double> rhs(cell_count * 9, 0.0);
        std::vector<double> coefficient(cell_count * 9, 0.0);
        std::fill(count, count + cell_count, 0.0);

        /* First reduction: mass, centroid, and constant-channel RHS. */
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            const std::int32_t label_value = labels[pixel];
            if (!valid_cell(label_value, cell_count)) {
                return BFFT_ERROR_INVALID_ARGUMENT;
            }
            const std::size_t cell =
                static_cast<std::size_t>(label_value);
            const std::size_t y = pixel / width;
            const std::size_t x = pixel - y * width;
            const double px =
                (static_cast<double>(x) + 0.5) /
                    static_cast<double>(width) - 0.5;
            const double py =
                (static_cast<double>(y) + 0.5) /
                    static_cast<double>(height) - 0.5;
            count[cell] += 1.0;
            centroid_sum[cell * 2] += px;
            centroid_sum[cell * 2 + 1] += py;
            for (std::size_t channel = 0; channel < 3; ++channel) {
                rhs[cell * 9 + channel] += target[pixel * 3 + channel];
            }
        }
        for (std::size_t cell = 0; cell < cell_count; ++cell) {
            count[cell] = std::max(count[cell], 1.0);
            centroid[cell * 2] =
                centroid_sum[cell * 2] / count[cell];
            centroid[cell * 2 + 1] =
                centroid_sum[cell * 2 + 1] / count[cell];
        }

        /* Second reduction: the physical covariance needed to set r. */
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            const std::size_t cell =
                static_cast<std::size_t>(labels[pixel]);
            const std::size_t y = pixel / width;
            const std::size_t x = pixel - y * width;
            const double px =
                (static_cast<double>(x) + 0.5) /
                    static_cast<double>(width) - 0.5;
            const double py =
                (static_cast<double>(y) + 0.5) /
                    static_cast<double>(height) - 0.5;
            const double dx = px - centroid[cell * 2];
            const double dy = py - centroid[cell * 2 + 1];
            second_moment[cell * 3] += dx * dx;
            second_moment[cell * 3 + 2] += dy * dy;
        }
        for (std::size_t cell = 0; cell < cell_count; ++cell) {
            radius[cell] = std::sqrt(std::max(
                (second_moment[cell * 3] +
                 second_moment[cell * 3 + 2]) / count[cell],
                1e-30));
        }
        std::fill(second_moment.begin(), second_moment.end(), 0.0);

        /*
           Third reduction: normalized slope block and slope-channel RHS.
           Writing the basis here lets the ridge rung reuse this exact frame.
        */
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            const std::size_t cell =
                static_cast<std::size_t>(labels[pixel]);
            const std::size_t y = pixel / width;
            const std::size_t x = pixel - y * width;
            const double px =
                (static_cast<double>(x) + 0.5) /
                    static_cast<double>(width) - 0.5;
            const double py =
                (static_cast<double>(y) + 0.5) /
                    static_cast<double>(height) - 0.5;
            const double ux =
                (px - centroid[cell * 2]) / radius[cell];
            const double uy =
                (py - centroid[cell * 2 + 1]) / radius[cell];
            basis[pixel * 3] = 1.0;
            basis[pixel * 3 + 1] = ux;
            basis[pixel * 3 + 2] = uy;
            second_moment[cell * 3] += ux * ux;
            second_moment[cell * 3 + 1] += ux * uy;
            second_moment[cell * 3 + 2] += uy * uy;
            for (std::size_t channel = 0; channel < 3; ++channel) {
                const double sample = target[pixel * 3 + channel];
                rhs[cell * 9 + 3 + channel] += ux * sample;
                rhs[cell * 9 + 6 + channel] += uy * sample;
            }
        }

        for (std::size_t cell = 0; cell < cell_count; ++cell) {
            const double gradient_regularization =
                1e-5 * count[cell] /
                std::max(radius[cell] * radius[cell], 1e-30);
            const double a =
                second_moment[cell * 3] + gradient_regularization;
            const double b = second_moment[cell * 3 + 1];
            const double c =
                second_moment[cell * 3 + 2] + gradient_regularization;
            const double determinant =
                std::max(a * c - b * b, 1e-30);
            for (std::size_t channel = 0; channel < 3; ++channel) {
                const double rhs0 = rhs[cell * 9 + channel];
                const double rhsx = rhs[cell * 9 + 3 + channel];
                const double rhsy = rhs[cell * 9 + 6 + channel];
                coefficient[cell * 9 + channel] =
                    rhs0 / ((1.0 + 1e-7) * count[cell]);
                coefficient[cell * 9 + 3 + channel] =
                    (c * rhsx - b * rhsy) / determinant;
                coefficient[cell * 9 + 6 + channel] =
                    (a * rhsy - b * rhsx) / determinant;
            }
        }

        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            const std::size_t cell =
                static_cast<std::size_t>(labels[pixel]);
            const double ux = basis[pixel * 3 + 1];
            const double uy = basis[pixel * 3 + 2];
            for (std::size_t channel = 0; channel < 3; ++channel) {
                reconstruction[pixel * 3 + channel] =
                    coefficient[cell * 9 + channel] +
                    ux * coefficient[cell * 9 + 3 + channel] +
                    uy * coefficient[cell * 9 + 6 + channel];
            }
        }
        return BFFT_OK;
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
}

bfft_status bfft_vision_hard_basis_refit(
    std::size_t pixel_count,
    std::size_t cell_count,
    std::size_t basis_width,
    const std::int32_t* labels,
    const double* design,
    const double* target,
    const double* count,
    const double* radius,
    double* reconstruction) {
    if (pixel_count == 0 || cell_count == 0 || basis_width < 3 ||
        labels == nullptr || design == nullptr || target == nullptr ||
        count == nullptr || radius == nullptr || reconstruction == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    std::size_t block_area = 0;
    std::size_t normal_size = 0;
    std::size_t rhs_width = 0;
    std::size_t rhs_size = 0;
    if (!checked_product(basis_width, basis_width, &block_area) ||
        !checked_product(cell_count, block_area, &normal_size) ||
        !checked_product(basis_width, std::size_t{3}, &rhs_width) ||
        !checked_product(cell_count, rhs_width, &rhs_size)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    try {
        std::vector<double> normal(normal_size, 0.0);
        std::vector<double> rhs(rhs_size, 0.0);
        std::vector<double> coefficient(rhs_size, 0.0);

        if (basis_width >= 12 && cell_count > 1) {
            /*
               A stable cell CSR exposes independent reductions without
               changing a single cell's summation order. The global raster
               interleaves cells; within each cell it is already ascending
               pixel order, exactly the order constructed here.
            */
            std::vector<std::size_t> offset(cell_count + 1, 0);
            for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
                const std::int32_t label_value = labels[pixel];
                if (!valid_cell(label_value, cell_count)) {
                    return BFFT_ERROR_INVALID_ARGUMENT;
                }
                ++offset[static_cast<std::size_t>(label_value) + 1];
            }
            for (std::size_t cell = 0; cell < cell_count; ++cell) {
                offset[cell + 1] += offset[cell];
            }
            std::vector<std::size_t> cursor(
                offset.begin(), offset.begin() + cell_count);
            std::vector<std::size_t> pixel_order(pixel_count);
            for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
                const std::size_t cell =
                    static_cast<std::size_t>(labels[pixel]);
                pixel_order[cursor[cell]++] = pixel;
            }
            const auto accumulate_cells = [&] (
                std::size_t begin, std::size_t end) {
                for (std::size_t cell = begin; cell < end; ++cell) {
                    double* block = normal.data() + cell * block_area;
                    double* cell_rhs = rhs.data() + cell * rhs_width;
                    for (std::size_t position = offset[cell];
                         position < offset[cell + 1]; ++position) {
                        const std::size_t pixel = pixel_order[position];
                        const double* row = design + pixel * basis_width;
                        for (std::size_t first = 0;
                             first < basis_width; ++first) {
                            const double u = row[first];
                            for (std::size_t second = first;
                                 second < basis_width; ++second) {
                                block[first * basis_width + second] +=
                                    u * row[second];
                            }
                            for (std::size_t channel = 0;
                                 channel < 3; ++channel) {
                                cell_rhs[first * 3 + channel] +=
                                    u * target[pixel * 3 + channel];
                            }
                        }
                    }
                }
            };
            const std::size_t mean_pixels =
                pixel_count / std::max(cell_count, std::size_t{1}) + 1;
            parallel_ranges(
                cell_count,
                mean_pixels * (block_area + rhs_width),
                0,
                accumulate_cells);
        } else {
            for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
                const std::int32_t label_value = labels[pixel];
                if (!valid_cell(label_value, cell_count)) {
                    return BFFT_ERROR_INVALID_ARGUMENT;
                }
                const std::size_t cell =
                    static_cast<std::size_t>(label_value);
                const double* row = design + pixel * basis_width;
                double* block = normal.data() + cell * block_area;
                double* cell_rhs = rhs.data() + cell * rhs_width;
                for (std::size_t first = 0;
                     first < basis_width; ++first) {
                    const double u = row[first];
                    for (std::size_t second = first;
                         second < basis_width; ++second) {
                        block[first * basis_width + second] +=
                            u * row[second];
                    }
                    for (std::size_t channel = 0;
                         channel < 3; ++channel) {
                        cell_rhs[first * 3 + channel] +=
                            u * target[pixel * 3 + channel];
                    }
                }
            }
        }

        std::atomic<int> solve_status{BFFT_OK};
        const auto solve_cells = [&](
            std::size_t begin, std::size_t end) {
            std::vector<double> factor(block_area, 0.0);
            std::vector<double> solve_rhs(rhs_width, 0.0);
            for (std::size_t cell = begin; cell < end; ++cell) {
                if (solve_status.load(std::memory_order_relaxed) != BFFT_OK) {
                    return;
                }
                double* block = normal.data() + cell * block_area;
                for (std::size_t row = 0; row < basis_width; ++row) {
                    for (std::size_t column = 0; column < row; ++column) {
                        block[row * basis_width + column] =
                            block[column * basis_width + row];
                    }
                }
                block[0] += 1e-7 * count[cell];
                const double gradient_regularization =
                    1e-5 * count[cell] /
                    std::max(radius[cell] * radius[cell], 1e-30);
                block[basis_width + 1] += gradient_regularization;
                block[2 * basis_width + 2] += gradient_regularization;
                for (std::size_t component = 3;
                     component < basis_width; ++component) {
                    block[component * basis_width + component] +=
                        2e-5 * count[cell];
                }

                const double* cell_rhs = rhs.data() + cell * rhs_width;
                double* cell_coefficient =
                    coefficient.data() + cell * rhs_width;
                std::copy(block, block + block_area, factor.begin());
                std::copy(
                    cell_rhs, cell_rhs + rhs_width, solve_rhs.begin());
                for (std::size_t column = 0;
                     column < basis_width; ++column) {
                    std::size_t pivot = column;
                    double pivot_size = std::abs(
                        factor[column * basis_width + column]);
                    for (std::size_t row = column + 1;
                         row < basis_width; ++row) {
                        const double candidate = std::abs(
                            factor[row * basis_width + column]);
                        if (candidate > pivot_size) {
                            pivot = row;
                            pivot_size = candidate;
                        }
                    }
                    if (!(pivot_size > 0.0) ||
                        !std::isfinite(pivot_size)) {
                        solve_status.store(
                            BFFT_ERROR_INTERNAL,
                            std::memory_order_relaxed);
                        return;
                    }
                    if (pivot != column) {
                        for (std::size_t inner = 0;
                             inner < basis_width; ++inner) {
                            std::swap(
                                factor[column * basis_width + inner],
                                factor[pivot * basis_width + inner]);
                        }
                        for (std::size_t channel = 0;
                             channel < 3; ++channel) {
                            std::swap(
                                solve_rhs[column * 3 + channel],
                                solve_rhs[pivot * 3 + channel]);
                        }
                    }
                    const double diagonal =
                        factor[column * basis_width + column];
                    for (std::size_t row = column + 1;
                         row < basis_width; ++row) {
                        const double multiplier =
                            factor[row * basis_width + column] / diagonal;
                        factor[row * basis_width + column] = multiplier;
                        for (std::size_t inner = column + 1;
                             inner < basis_width; ++inner) {
                            factor[row * basis_width + inner] -=
                                multiplier *
                                factor[column * basis_width + inner];
                        }
                        for (std::size_t channel = 0;
                             channel < 3; ++channel) {
                            solve_rhs[row * 3 + channel] -=
                                multiplier *
                                solve_rhs[column * 3 + channel];
                        }
                    }
                }
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    for (std::size_t reverse = basis_width;
                         reverse-- > 0;) {
                        double value = solve_rhs[reverse * 3 + channel];
                        for (std::size_t inner = reverse + 1;
                             inner < basis_width; ++inner) {
                            value -=
                                factor[reverse * basis_width + inner] *
                                cell_coefficient[inner * 3 + channel];
                        }
                        cell_coefficient[reverse * 3 + channel] =
                            value /
                            factor[reverse * basis_width + reverse];
                    }
                }
            }
        };
        parallel_ranges(
            cell_count, block_area + rhs_width, 0, solve_cells);
        if (solve_status.load(std::memory_order_relaxed) != BFFT_OK) {
            return static_cast<bfft_status>(
                solve_status.load(std::memory_order_relaxed));
        }
        const auto render_pixels = [&] (
            std::size_t begin, std::size_t end) {
            for (std::size_t pixel = begin; pixel < end; ++pixel) {
                const std::size_t cell =
                    static_cast<std::size_t>(labels[pixel]);
                const double* row = design + pixel * basis_width;
                const double* cell_coefficient =
                    coefficient.data() + cell * rhs_width;
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    double value = 0.0;
                    for (std::size_t component = 0;
                         component < basis_width; ++component) {
                        value += row[component] *
                            cell_coefficient[component * 3 + channel];
                    }
                    reconstruction[pixel * 3 + channel] = value;
                }
            }
        };
        parallel_ranges(
            pixel_count, basis_width * 3, 0, render_pixels);
        return BFFT_OK;
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
}

bfft_status bfft_vision_separable_filter_f64(
    std::size_t channels,
    std::size_t height,
    std::size_t width,
    std::size_t kernel_y_size,
    std::size_t kernel_x_size,
    std::uint8_t mirror_without_edge,
    std::size_t thread_count,
    const double* fields,
    const double* kernel_y,
    const double* kernel_x,
    double* scratch,
    double* output) {
    if (channels == 0 || height == 0 || width == 0 ||
        kernel_y_size == 0 || kernel_x_size == 0 ||
        kernel_y_size % 2 == 0 || kernel_x_size % 2 == 0 ||
        height > static_cast<std::size_t>(
            std::numeric_limits<std::ptrdiff_t>::max()) ||
        width > static_cast<std::size_t>(
            std::numeric_limits<std::ptrdiff_t>::max()) ||
        fields == nullptr || kernel_y == nullptr || kernel_x == nullptr ||
        scratch == nullptr || output == nullptr || fields == scratch ||
        fields == output || scratch == output) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    std::size_t row_tasks = 0;
    std::size_t column_tasks = 0;
    std::size_t values = 0;
    if (!checked_product(channels, height, &row_tasks) ||
        !checked_product(channels, width, &column_tasks) ||
        !checked_product(row_tasks, width, &values)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    (void)values;
    const std::ptrdiff_t radius_x =
        static_cast<std::ptrdiff_t>(kernel_x_size / 2);
    const std::ptrdiff_t radius_y =
        static_cast<std::ptrdiff_t>(kernel_y_size / 2);
    const bool whole_sample = mirror_without_edge != 0;
    try {
        parallel_ranges(
            row_tasks, width, thread_count,
            [&](std::size_t begin, std::size_t end) {
                for (std::size_t row = begin; row < end; ++row) {
                    const std::size_t channel = row / height;
                    const std::size_t y = row - channel * height;
                    const std::size_t base = (channel * height + y) * width;
                    for (std::size_t x = 0; x < width; ++x) {
                        double value = 0.0;
                        for (std::ptrdiff_t offset = -radius_x;
                             offset <= radius_x; ++offset) {
                            const std::size_t source_x = reflected_index(
                                static_cast<std::ptrdiff_t>(x) + offset,
                                width, whole_sample);
                            value += kernel_x[
                                static_cast<std::size_t>(offset + radius_x)] *
                                fields[base + source_x];
                        }
                        scratch[base + x] = value;
                    }
                }
            });
        parallel_ranges(
            column_tasks, height, thread_count,
            [&](std::size_t begin, std::size_t end) {
                for (std::size_t column = begin; column < end; ++column) {
                    const std::size_t channel = column / width;
                    const std::size_t x = column - channel * width;
                    for (std::size_t y = 0; y < height; ++y) {
                        double value = 0.0;
                        for (std::ptrdiff_t offset = -radius_y;
                             offset <= radius_y; ++offset) {
                            const std::size_t source_y = reflected_index(
                                static_cast<std::ptrdiff_t>(y) + offset,
                                height, whole_sample);
                            value += kernel_y[
                                static_cast<std::size_t>(offset + radius_y)] *
                                scratch[
                                    (channel * height + source_y) * width + x];
                        }
                        output[(channel * height + y) * width + x] = value;
                    }
                }
            });
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
    return BFFT_OK;
}

bfft_status bfft_vision_resize_bilinear_f64(
    std::size_t channels,
    std::size_t input_height,
    std::size_t input_width,
    std::size_t output_height,
    std::size_t output_width,
    std::size_t thread_count,
    const double* fields,
    double* output) {
    if (channels == 0 || input_height == 0 || input_width == 0 ||
        output_height == 0 || output_width == 0 ||
        input_height > static_cast<std::size_t>(
            std::numeric_limits<std::ptrdiff_t>::max()) ||
        input_width > static_cast<std::size_t>(
            std::numeric_limits<std::ptrdiff_t>::max()) ||
        fields == nullptr || output == nullptr || fields == output) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    std::size_t input_planes = 0;
    std::size_t output_planes = 0;
    std::size_t input_values = 0;
    std::size_t output_values = 0;
    std::size_t row_tasks = 0;
    if (!checked_product(input_height, input_width, &input_planes) ||
        !checked_product(output_height, output_width, &output_planes) ||
        !checked_product(channels, input_planes, &input_values) ||
        !checked_product(channels, output_planes, &output_values) ||
        !checked_product(channels, output_height, &row_tasks)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    (void)input_values;
    (void)output_values;
    const double scale_y =
        static_cast<double>(input_height) / output_height;
    const double scale_x =
        static_cast<double>(input_width) / output_width;
    try {
        parallel_ranges(
            row_tasks, output_width, thread_count,
            [&](std::size_t begin, std::size_t end) {
                for (std::size_t row = begin; row < end; ++row) {
                    const std::size_t channel = row / output_height;
                    const std::size_t y = row - channel * output_height;
                    const double source_y = (y + 0.5) * scale_y - 0.5;
                    const auto raw_y0 = static_cast<std::ptrdiff_t>(
                        std::floor(source_y));
                    const double fraction_y = source_y - raw_y0;
                    const std::size_t y0 = reflected_index(
                        raw_y0, input_height, true);
                    const std::size_t y1 = reflected_index(
                        raw_y0 + 1, input_height, true);
                    const double* plane = fields + channel * input_planes;
                    double* target = output + channel * output_planes +
                        y * output_width;
                    for (std::size_t x = 0; x < output_width; ++x) {
                        const double source_x =
                            (x + 0.5) * scale_x - 0.5;
                        const auto raw_x0 = static_cast<std::ptrdiff_t>(
                            std::floor(source_x));
                        const double fraction_x = source_x - raw_x0;
                        const std::size_t x0 = reflected_index(
                            raw_x0, input_width, true);
                        const std::size_t x1 = reflected_index(
                            raw_x0 + 1, input_width, true);
                        const double top =
                            (1.0 - fraction_x) * plane[y0 * input_width + x0] +
                            fraction_x * plane[y0 * input_width + x1];
                        const double bottom =
                            (1.0 - fraction_x) * plane[y1 * input_width + x0] +
                            fraction_x * plane[y1 * input_width + x1];
                        target[x] =
                            (1.0 - fraction_y) * top + fraction_y * bottom;
                    }
                }
            });
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
    return BFFT_OK;
}

bfft_status bfft_vision_sobel_f64(
    std::size_t channels,
    std::size_t height,
    std::size_t width,
    std::size_t thread_count,
    const double* fields,
    double* gradient_x,
    double* gradient_y) {
    if (channels == 0 || height == 0 || width == 0 || fields == nullptr ||
        gradient_x == nullptr || gradient_y == nullptr ||
        fields == gradient_x || fields == gradient_y ||
        gradient_x == gradient_y) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    std::size_t plane = 0;
    std::size_t values = 0;
    std::size_t row_tasks = 0;
    if (!checked_product(height, width, &plane) ||
        !checked_product(channels, plane, &values) ||
        !checked_product(channels, height, &row_tasks)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    (void)values;
    try {
        parallel_ranges(
            row_tasks, width, thread_count,
            [&](std::size_t begin, std::size_t end) {
                for (std::size_t row = begin; row < end; ++row) {
                    const std::size_t channel = row / height;
                    const std::size_t y = row - channel * height;
                    const std::size_t above = y == 0 ? 0 : y - 1;
                    const std::size_t below =
                        std::min(y + 1, height - 1);
                    const double* source = fields + channel * plane;
                    double* gx = gradient_x + channel * plane + y * width;
                    double* gy = gradient_y + channel * plane + y * width;
                    for (std::size_t x = 0; x < width; ++x) {
                        const std::size_t left = x == 0 ? 0 : x - 1;
                        const std::size_t right =
                            std::min(x + 1, width - 1);
                        gx[x] = (
                            source[above * width + right] +
                            2.0 * source[y * width + right] +
                            source[below * width + right] -
                            source[above * width + left] -
                            2.0 * source[y * width + left] -
                            source[below * width + left]) / 8.0;
                        gy[x] = (
                            source[below * width + left] +
                            2.0 * source[below * width + x] +
                            source[below * width + right] -
                            source[above * width + left] -
                            2.0 * source[above * width + x] -
                            source[above * width + right]) / 8.0;
                    }
                }
            });
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
    return BFFT_OK;
}

bfft_status bfft_vision_nearest_code_f32(
    std::size_t observation_count,
    std::size_t code_count,
    std::size_t dimensions,
    std::size_t thread_count,
    const float* observations,
    const float* codes,
    std::int32_t* labels) {
    if (observation_count == 0 || code_count == 0 || dimensions == 0 ||
        code_count > static_cast<std::size_t>(
            std::numeric_limits<std::int32_t>::max()) ||
        observations == nullptr || codes == nullptr || labels == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    std::size_t observation_values = 0;
    std::size_t code_values = 0;
    if (!checked_product(observation_count, dimensions, &observation_values) ||
        !checked_product(code_count, dimensions, &code_values)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    (void)observation_values;
    (void)code_values;
    try {
        parallel_ranges(
            observation_count, code_count * dimensions, thread_count,
            [&](std::size_t begin, std::size_t end) {
                for (std::size_t observation = begin;
                     observation < end; ++observation) {
                    const float* value =
                        observations + observation * dimensions;
                    float best_distance =
                        std::numeric_limits<float>::infinity();
                    std::int32_t best_code = 0;
                    for (std::size_t code = 0; code < code_count; ++code) {
                        const float* center = codes + code * dimensions;
                        float distance = 0.0F;
                        for (std::size_t dimension = 0;
                             dimension < dimensions; ++dimension) {
                            const float delta = value[dimension] -
                                center[dimension];
                            distance += delta * delta;
                        }
                        if (distance < best_distance) {
                            best_distance = distance;
                            best_code = static_cast<std::int32_t>(code);
                        }
                    }
                    labels[observation] = best_code;
                }
            });
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
    return BFFT_OK;
}

bfft_status bfft_vision_weighted_kmeanspp_f32(
    std::size_t observation_count,
    std::size_t code_count,
    std::size_t dimensions,
    const float* observations,
    const double* weights,
    const double* random_draws,
    float* codes,
    std::size_t* live_code_count) {
    if (observation_count == 0 || code_count == 0 || dimensions == 0 ||
        observations == nullptr || weights == nullptr ||
        random_draws == nullptr || codes == nullptr ||
        live_code_count == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    std::size_t observation_values = 0;
    std::size_t code_values = 0;
    if (!checked_product(observation_count, dimensions, &observation_values) ||
        !checked_product(code_count, dimensions, &code_values)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    (void)observation_values;
    (void)code_values;
    try {
        std::vector<float> closest(observation_count);
        double total = 0.0;
        for (std::size_t observation = 0; observation < observation_count;
             ++observation) {
            if (!std::isfinite(weights[observation]) ||
                weights[observation] < 0.0) {
                return BFFT_ERROR_INVALID_ARGUMENT;
            }
            total += weights[observation];
        }
        if (!(total > 0.0)) {
            return BFFT_ERROR_INVALID_ARGUMENT;
        }
        auto choose = [&](double draw, bool weighted_distance) {
            double cumulative = 0.0;
            std::size_t choice = observation_count - 1;
            for (std::size_t observation = 0;
                 observation < observation_count; ++observation) {
                const double probability = weighted_distance
                    ? weights[observation] *
                        static_cast<double>(closest[observation])
                    : weights[observation];
                cumulative += probability / total;
                if (draw < cumulative) {
                    choice = observation;
                    break;
                }
            }
            return choice;
        };
        std::size_t choice = choose(random_draws[0], false);
        std::copy(
            observations + choice * dimensions,
            observations + (choice + 1) * dimensions,
            codes);
        for (std::size_t observation = 0; observation < observation_count;
             ++observation) {
            float distance = 0.0F;
            for (std::size_t dimension = 0; dimension < dimensions;
                 ++dimension) {
                const float delta =
                    observations[observation * dimensions + dimension] -
                    codes[dimension];
                distance += delta * delta;
            }
            closest[observation] = distance;
        }
        std::size_t live = 1;
        for (std::size_t code = 1; code < code_count; ++code) {
            total = 0.0;
            for (std::size_t observation = 0;
                 observation < observation_count; ++observation) {
                total += weights[observation] *
                    static_cast<double>(closest[observation]);
            }
            if (!(total > 1e-12)) {
                break;
            }
            choice = choose(random_draws[code], true);
            float* center = codes + code * dimensions;
            std::copy(
                observations + choice * dimensions,
                observations + (choice + 1) * dimensions,
                center);
            for (std::size_t observation = 0;
                 observation < observation_count; ++observation) {
                float distance = 0.0F;
                for (std::size_t dimension = 0; dimension < dimensions;
                     ++dimension) {
                    const float delta =
                        observations[observation * dimensions + dimension] -
                        center[dimension];
                    distance += delta * delta;
                }
                closest[observation] = std::min(
                    closest[observation], distance);
            }
            live = code + 1;
        }
        *live_code_count = live;
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
    return BFFT_OK;
}

bfft_status bfft_vision_block_dct8_f64(
    std::size_t height,
    std::size_t width,
    std::size_t thread_count,
    const double* input,
    const double* matrix,
    double* coefficients) {
    if (height == 0 || width == 0 || input == nullptr || matrix == nullptr ||
        coefficients == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    const std::size_t block_height = (height + 7) / 8;
    const std::size_t block_width = (width + 7) / 8;
    std::size_t block_count = 0;
    if (!checked_product(block_height, block_width, &block_count)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    try {
        parallel_ranges(
            block_count, std::size_t{1024}, thread_count,
            [&](std::size_t begin, std::size_t end) {
                for (std::size_t block = begin; block < end; ++block) {
                    const std::size_t block_y = block / block_width;
                    const std::size_t block_x = block % block_width;
                    double temporary[64];
                    for (std::size_t u = 0; u < 8; ++u) {
                        for (std::size_t j = 0; j < 8; ++j) {
                            double sum = 0.0;
                            const std::size_t y_base = block_y * 8;
                            const std::size_t x = std::min(
                                block_x * 8 + j, width - 1);
                            for (std::size_t i = 0; i < 8; ++i) {
                                const std::size_t y = std::min(
                                    y_base + i, height - 1);
                                sum += matrix[u * 8 + i] *
                                    (input[y * width + x] - 128.0);
                            }
                            temporary[u * 8 + j] = sum;
                        }
                    }
                    double* output = coefficients + block * 64;
                    for (std::size_t u = 0; u < 8; ++u) {
                        for (std::size_t v = 0; v < 8; ++v) {
                            double sum = 0.0;
                            for (std::size_t j = 0; j < 8; ++j) {
                                sum += temporary[u * 8 + j] *
                                    matrix[v * 8 + j];
                            }
                            output[u * 8 + v] = sum;
                        }
                    }
                }
            });
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
    return BFFT_OK;
}

bfft_status bfft_vision_inverse_block_dct8_f64(
    std::size_t height,
    std::size_t width,
    std::size_t thread_count,
    const double* coefficients,
    const double* matrix,
    double* output) {
    if (height == 0 || width == 0 || coefficients == nullptr ||
        matrix == nullptr || output == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    const std::size_t block_height = (height + 7) / 8;
    const std::size_t block_width = (width + 7) / 8;
    std::size_t block_count = 0;
    if (!checked_product(block_height, block_width, &block_count)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    try {
        parallel_ranges(
            block_count, std::size_t{1024}, thread_count,
            [&](std::size_t begin, std::size_t end) {
                for (std::size_t block = begin; block < end; ++block) {
                    const std::size_t block_y = block / block_width;
                    const std::size_t block_x = block % block_width;
                    const double* source = coefficients + block * 64;
                    double temporary[64];
                    for (std::size_t i = 0; i < 8; ++i) {
                        for (std::size_t v = 0; v < 8; ++v) {
                            double sum = 0.0;
                            for (std::size_t u = 0; u < 8; ++u) {
                                sum += matrix[u * 8 + i] *
                                    source[u * 8 + v];
                            }
                            temporary[i * 8 + v] = sum;
                        }
                    }
                    for (std::size_t i = 0; i < 8; ++i) {
                        const std::size_t y = block_y * 8 + i;
                        if (y >= height) {
                            continue;
                        }
                        for (std::size_t j = 0; j < 8; ++j) {
                            const std::size_t x = block_x * 8 + j;
                            if (x >= width) {
                                continue;
                            }
                            double sum = 0.0;
                            for (std::size_t v = 0; v < 8; ++v) {
                                sum += temporary[i * 8 + v] *
                                    matrix[v * 8 + j];
                            }
                            output[y * width + x] = sum + 128.0;
                        }
                    }
                }
            });
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
    return BFFT_OK;
}

bfft_status bfft_vision_image_metrics_f64(
    std::size_t height,
    std::size_t width,
    std::size_t kernel_size,
    std::size_t thread_count,
    const double* reference,
    const double* candidate,
    const double* reference_mean,
    const double* reference_variance,
    const double* reference_edge,
    const double* kernel,
    double* scratch,
    double* metrics) {
    if (height == 0 || width == 0 || kernel_size == 0 ||
        kernel_size % 2 == 0 || reference == nullptr ||
        candidate == nullptr || reference_mean == nullptr ||
        reference_variance == nullptr || reference_edge == nullptr ||
        kernel == nullptr || scratch == nullptr || metrics == nullptr) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    std::size_t pixels = 0;
    std::size_t scratch_size = 0;
    if (!checked_product(height, width, &pixels) ||
        !checked_product(pixels, std::size_t{9}, &scratch_size)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    (void)scratch_size;
    const std::ptrdiff_t radius =
        static_cast<std::ptrdiff_t>(kernel_size / 2);
    try {
        // One branch per output pixel selects the boundary or branch-free
        // interior path; the kernel tap loop itself has no conditionals.
        parallel_ranges(
            height, width, thread_count,
            [&](std::size_t begin, std::size_t end) {
                for (std::size_t y = begin; y < end; ++y) {
                    for (std::size_t x = 0; x < width; ++x) {
                        const bool interior =
                            x >= static_cast<std::size_t>(radius) &&
                            x + static_cast<std::size_t>(radius) < width;
                        double* destination = scratch + (y * width + x) * 9;
                        for (std::size_t channel = 0; channel < 3; ++channel) {
                            double mean = 0.0;
                            double square = 0.0;
                            double product = 0.0;
                            for (std::ptrdiff_t offset = -radius;
                                 offset <= radius; ++offset) {
                                const std::size_t source_x = interior
                                    ? static_cast<std::size_t>(
                                        static_cast<std::ptrdiff_t>(x) + offset)
                                    : reflected_index(
                                        static_cast<std::ptrdiff_t>(x) + offset,
                                        width, false);
                                const std::size_t source =
                                    (y * width + source_x) * 3 + channel;
                                const double weight = kernel[
                                    static_cast<std::size_t>(offset + radius)];
                                const double value = candidate[source];
                                mean += weight * value;
                                square += weight * value * value;
                                product += weight * reference[source] * value;
                            }
                            destination[channel * 3] = mean;
                            destination[channel * 3 + 1] = square;
                            destination[channel * 3 + 2] = product;
                        }
                    }
                }
            });

        std::vector<double> row_mse(height, 0.0);
        std::vector<double> row_ssim(height, 0.0);
        std::vector<double> row_edge(height, 0.0);
        constexpr double c1 = 6.5025;
        constexpr double c2 = 58.5225;
        auto luma = [&](std::size_t y, std::size_t x) {
            const double* rgb = candidate + (y * width + x) * 3;
            return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
        };
        parallel_ranges(
            height, width, thread_count,
            [&](std::size_t begin, std::size_t end) {
                for (std::size_t y = begin; y < end; ++y) {
                    double mse_sum = 0.0;
                    double ssim_sum = 0.0;
                    double edge_sum = 0.0;
                    const bool interior_y =
                        y >= static_cast<std::size_t>(radius) &&
                        y + static_cast<std::size_t>(radius) < height;
                    const std::size_t above = y == 0 ? 0 : y - 1;
                    const std::size_t below = std::min(y + 1, height - 1);
                    for (std::size_t x = 0; x < width; ++x) {
                        const std::size_t pixel = y * width + x;
                        for (std::size_t channel = 0; channel < 3; ++channel) {
                            double mean = 0.0;
                            double square = 0.0;
                            double product = 0.0;
                            for (std::ptrdiff_t offset = -radius;
                                 offset <= radius; ++offset) {
                                const std::size_t source_y = interior_y
                                    ? static_cast<std::size_t>(
                                        static_cast<std::ptrdiff_t>(y) + offset)
                                    : reflected_index(
                                        static_cast<std::ptrdiff_t>(y) + offset,
                                        height, false);
                                const double* source = scratch +
                                    (source_y * width + x) * 9 + channel * 3;
                                const double weight = kernel[
                                    static_cast<std::size_t>(offset + radius)];
                                mean += weight * source[0];
                                square += weight * source[1];
                                product += weight * source[2];
                            }
                            const std::size_t rgb_index = pixel * 3 + channel;
                            const double source_mean = reference_mean[rgb_index];
                            const double source_variance =
                                reference_variance[rgb_index];
                            const double candidate_variance = std::max(
                                square - mean * mean, 0.0);
                            const double covariance =
                                product - source_mean * mean;
                            const double numerator =
                                (2.0 * source_mean * mean + c1) *
                                (2.0 * covariance + c2);
                            const double denominator =
                                (source_mean * source_mean + mean * mean + c1) *
                                (source_variance + candidate_variance + c2) +
                                1e-20;
                            ssim_sum += numerator / denominator;
                            const double difference =
                                reference[rgb_index] - candidate[rgb_index];
                            mse_sum += difference * difference;
                        }
                        const std::size_t left = x == 0 ? 0 : x - 1;
                        const std::size_t right = std::min(x + 1, width - 1);
                        const double axis0 =
                            luma(below, left) + 2.0 * luma(below, x) +
                            luma(below, right) - luma(above, left) -
                            2.0 * luma(above, x) - luma(above, right);
                        const double axis1 =
                            luma(above, right) + 2.0 * luma(y, right) +
                            luma(below, right) - luma(above, left) -
                            2.0 * luma(y, left) - luma(below, left);
                        const double edge0 =
                            reference_edge[pixel * 2] - axis0;
                        const double edge1 =
                            reference_edge[pixel * 2 + 1] - axis1;
                        edge_sum += edge0 * edge0 + edge1 * edge1;
                    }
                    row_mse[y] = mse_sum;
                    row_ssim[y] = ssim_sum;
                    row_edge[y] = edge_sum;
                }
            });
        double mse_sum = 0.0;
        double ssim_sum = 0.0;
        double edge_sum = 0.0;
        for (std::size_t y = 0; y < height; ++y) {
            mse_sum += row_mse[y];
            ssim_sum += row_ssim[y];
            edge_sum += row_edge[y];
        }
        metrics[0] = mse_sum / static_cast<double>(pixels * 3);
        metrics[1] = ssim_sum / static_cast<double>(pixels * 3);
        metrics[2] = edge_sum / static_cast<double>(pixels * 2);
        return BFFT_OK;
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
}

bfft_status bfft_vision_binary_dilation_cross_u8(
    std::size_t height,
    std::size_t width,
    std::size_t iterations,
    std::size_t thread_count,
    const std::uint8_t* mask,
    std::uint8_t* scratch,
    std::uint8_t* output) {
    if (height == 0 || width == 0 || mask == nullptr || scratch == nullptr ||
        output == nullptr || mask == scratch || mask == output ||
        scratch == output) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    std::size_t pixels = 0;
    if (!checked_product(height, width, &pixels)) {
        return BFFT_ERROR_INVALID_ARGUMENT;
    }
    std::copy(mask, mask + pixels, output);
    std::uint8_t* source = output;
    std::uint8_t* target = scratch;
    try {
        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            parallel_ranges(
                height, width, thread_count,
                [&](std::size_t begin, std::size_t end) {
                    for (std::size_t y = begin; y < end; ++y) {
                        for (std::size_t x = 0; x < width; ++x) {
                            const std::size_t p = y * width + x;
                            target[p] = static_cast<std::uint8_t>(
                                source[p] != 0 ||
                                (y > 0 && source[p - width] != 0) ||
                                (y + 1 < height && source[p + width] != 0) ||
                                (x > 0 && source[p - 1] != 0) ||
                                (x + 1 < width && source[p + 1] != 0));
                        }
                    }
                });
            std::swap(source, target);
        }
        if (source != output) {
            std::copy(source, source + pixels, output);
        }
    } catch (const std::bad_alloc&) {
        return BFFT_ERROR_ALLOCATION;
    } catch (...) {
        return BFFT_ERROR_INTERNAL;
    }
    return BFFT_OK;
}

}  // extern "C"
