#ifndef BFFT_VISION_H
#define BFFT_VISION_H

#include <bfft/bfft.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
   Exact kernels for a measured two-owner partition-of-unity image model.

   These routines do not search, generate, or reuse candidate graphs.  The
   caller supplies the owner/runner assignment measured for the current image
   state and the block slots of that one graph.

   Arrays are contiguous and row-major.  The image model has a caller-selected
   basis width and exactly three output channels.
*/

/*
   Assemble G = A^T A and h = A^T target directly, without materializing A.

   For pixel p, let

       u = owner_weight[p]  * owner_basis[p, :]
       v = runner_weight[p] * runner_basis[p, :].

   The pixel contributes uu^T to its owner diagonal block, and, when
   has_runner[p] is nonzero, vv^T to the runner diagonal block plus uv^T and
   vu^T to the supplied directed off-diagonal block slots.  The right-hand
   side receives u*target and v*target for each of three channels.

   owner and runner contain cell indices in [0, cell_count).  runner must be a
   valid fallback index even when has_runner is zero.  diagonal_block has
   cell_count entries.  owner_runner_block and runner_owner_block have
   pixel_count entries; their values are read only where has_runner is nonzero.

   normal_blocks is overwritten and has
       normal_block_count * basis_width * basis_width doubles.
   rhs is overwritten and has
       cell_count * basis_width * 3 doubles.
*/
bfft_status bfft_vision_assemble_normal(
    size_t pixel_count,
    size_t cell_count,
    size_t basis_width,
    size_t normal_block_count,
    const int32_t* owner,
    const int32_t* runner,
    const uint8_t* has_runner,
    const double* owner_weight,
    const double* runner_weight,
    const double* owner_basis,
    const double* runner_basis,
    const double* target,
    const int64_t* diagonal_block,
    const int64_t* owner_runner_block,
    const int64_t* runner_owner_block,
    double* normal_blocks,
    double* rhs);

/*
   Render the two affine predictions and their partition-of-unity blend:

       owner_prediction[p, c] =
           dot(coeff[owner[p], :, c], owner_basis[p, :])
       runner_prediction[p, c] =
           dot(coeff[runner[p], :, c], runner_basis[p, :])
       field[p, c] =
           owner_weight[p] * owner_prediction[p, c]
         + runner_weight[p] * runner_prediction[p, c].

   coeff has cell_count * basis_width * 3 doubles.  Each output is overwritten
   and has pixel_count * 3 doubles.  Output arrays must not overlap inputs or
   one another.
*/
bfft_status bfft_vision_render_affine(
    size_t pixel_count,
    size_t cell_count,
    size_t basis_width,
    const int32_t* owner,
    const int32_t* runner,
    const double* owner_weight,
    const double* runner_weight,
    const double* owner_basis,
    const double* runner_basis,
    const double* coeff,
    double* owner_prediction,
    double* runner_prediction,
    double* field);

/*
   Find the best bounded residual ridge independently inside every measured
   owner cell.  For each supplied angle and threshold bin, the score is the
   channel-weighted squared signed residual sum divided by cell mass.  The
   first angle and first bin win exact ties.

   residual is pixel_count x 3.  dx and dy are pixel offsets from their
   measured owner site.  angle_cos and angle_sin have angle_count entries.
   channel_weight has three entries.  score, best_angle, and best_bin each
   have cell_count entries and are overwritten.
*/
bfft_status bfft_vision_scan_residual_ridges(
    size_t pixel_count,
    size_t cell_count,
    size_t angle_count,
    size_t bin_count,
    double spacing,
    double span,
    const int32_t* owner,
    const double* pixel_weight,
    const double* residual,
    const double* dx,
    const double* dy,
    const double* angle_cos,
    const double* angle_sin,
    const double* channel_weight,
    double* score,
    int32_t* best_angle,
    int32_t* best_bin);

/*
   Find the best split offset along one premeasured coordinate per cell.

   projection contains the normalized signed coordinate of every pixel in its
   owner's frame. Pixels are counting-sorted by owner once; each cell then
   reuses one bin_count x 3 histogram. This avoids the
   cell_count x angle_count x bin_count accumulator of the free-angle control.

   The score and tie rule match bfft_vision_scan_residual_ridges. score and
   best_bin each have cell_count entries and are overwritten.
*/
bfft_status bfft_vision_scan_paired_offsets(
    size_t pixel_count,
    size_t cell_count,
    size_t bin_count,
    double span,
    const int32_t* owner,
    const double* pixel_weight,
    const double* residual,
    const double* projection,
    const double* channel_weight,
    double* score,
    int32_t* best_bin);

/* Compact multi-support affine operator used by the HD matrix-free solver.
   Each sample contributes

       weight * (c0 + basis_x*c1 + basis_y*c2)

   from one cell to one pixel.  rows and sites have sample_count entries.
   The normal application overwrites both pixel_scratch[pixel_count] and
   output[3*cell_count]. */
bfft_status bfft_vision_support_forward(
    size_t sample_count, size_t pixel_count, size_t cell_count,
    const int32_t* rows, const int32_t* sites, const double* weight,
    const double* basis_x, const double* basis_y, const double* coefficient,
    double* pixel);

bfft_status bfft_vision_support_transpose(
    size_t sample_count, size_t pixel_count, size_t cell_count,
    const int32_t* rows, const int32_t* sites, const double* weight,
    const double* basis_x, const double* basis_y, const double* pixel,
    double* coefficient);

bfft_status bfft_vision_support_normal_apply(
    size_t sample_count, size_t pixel_count, size_t cell_count,
    const int32_t* rows, const int32_t* sites, const double* weight,
    const double* basis_x, const double* basis_y, const double* coefficient,
    double* pixel_scratch, double* output);

/*
   Correct tensor determinant density when a locally anisotropic support turns
   too far to remain straight.

   precision_* and base_measure are height*width float32 images. base_measure
   is normalized and base_implied_cells restores its physical population
   scale. All four image outputs are overwritten. corrected_measure is
   normalized; corrected_implied_cells receives its physical integral.

   The doubled-angle director derivative is sign invariant. The correction is

       sqrt(max(1, kappa*a*a/(2*b))),

   where a and b are the tangent and normal tensor semi-spans.
*/
bfft_status bfft_vision_curvature_population_f32(
    size_t height, size_t width,
    const float* precision_xx, const float* precision_xy,
    const float* precision_yy, const float* base_measure,
    double base_implied_cells,
    float* corrected_measure, float* director_curvature,
    float* sagitta_ratio, float* population_factor,
    double* corrected_implied_cells);

/*
   Diffuse an HxWxC field through four undirected conductance families.

   horizontal is Hx(W-1), vertical is (H-1)xW, and each diagonal family is
   (H-1)x(W-1). output and scratch each contain H*W*C doubles and must not
   overlap one another or the inputs. Every step is a convex gather, preserving
   constants and the partition-of-unity sum exactly up to roundoff.
*/
bfft_status bfft_vision_soft_support_diffuse(
    size_t height, size_t width, size_t channels, size_t passes,
    size_t thread_count,
    double coupling, const double* field,
    const double* horizontal, const double* vertical,
    const double* diagonal_down_right, const double* diagonal_down_left,
    double* output, double* scratch);

/*
   Prepare the fixed receiver-local data for one continuous FM-LBR march.
   superbase is HxWx3x2. The routine emits cyclic HxWx6x2 directions,
   integrated direction costs/validity, HxWx4 cardinal costs, and the exact
   accepted-vertex-to-receiver CSR used by the marcher. inverse_receiver has
   inverse_capacity entries; 10*H*W is always sufficient.
*/
bfft_status bfft_vision_prepare_continuous_metric(
    size_t height, size_t width, double consistency_limit,
    const int32_t* superbase,
    const double* mxx, const double* mxy, const double* myy,
    int32_t* directions, double* direction_costs,
    uint8_t* direction_valid, double* cardinal_costs,
    int64_t* inverse_offset, size_t inverse_capacity,
    int32_t* inverse_receiver, size_t* inverse_count);

/*
   Exact one-label anisotropic fast march used by the continuous transport
   partition. The receiver-local stencil has six cyclic directions and four
   cardinal connectivity edges. inverse_offset/inverse_receiver is the CSR
   inverse incidence from an accepted vertex to every receiver using it.

   The implementation keeps one decrease-key heap entry per pixel. Outputs
   preserve the reference walk's owner, distance, covector, source covector,
   parent simplex, and acceptance-order bookkeeping.

   directions is HxWx6x2, direction_costs and direction_valid are HxWx6,
   cardinal_costs is HxWx4, and metric components are HxW. accepted_count,
   push_count, and maximum_heap_size are scalar outputs.
*/
bfft_status bfft_vision_fast_march_first_label(
    size_t height, size_t width, size_t seed_count,
    const int32_t* seed_pixel, const double* seed_value,
    const int32_t* seed_label, const double* seed_gradient_x,
    const double* seed_gradient_y, const int32_t* directions,
    const double* direction_costs, const uint8_t* direction_valid,
    const double* cardinal_costs, const int64_t* inverse_offset,
    size_t inverse_count, const int32_t* inverse_receiver,
    const double* mxx, const double* mxy, const double* myy,
    int32_t* owner, double* distance, double* gradient_x,
    double* gradient_y, double* source_gradient_x,
    double* source_gradient_y, int32_t* parent_first,
    int32_t* parent_second, double* parent_fraction,
    int32_t* acceptance_order, size_t* accepted_count,
    size_t* push_count, size_t* maximum_heap_size);

/*
   Exact owner/distance-only form of bfft_vision_fast_march_first_label.
   The recurrence and decrease-key acceptance order are unchanged; covector,
   parent-simplex, and acceptance-order output streams are omitted.
*/
bfft_status bfft_vision_fast_march_labels(
    size_t height, size_t width, size_t seed_count,
    const int32_t* seed_pixel, const double* seed_value,
    const int32_t* seed_label, const double* seed_gradient_x,
    const double* seed_gradient_y, const int32_t* directions,
    const double* direction_costs, const uint8_t* direction_valid,
    const double* cardinal_costs, const int64_t* inverse_offset,
    size_t inverse_count, const int32_t* inverse_receiver,
    const double* mxx, const double* mxy, const double* myy,
    int32_t* owner, double* distance,
    size_t* push_count, size_t* maximum_heap_size);

/*
   Assemble the direction-major float32 eight-neighbour cost stack directly
   from frozen precision and optional boundary tensors. precision_gain is the
   already-scaled metric coefficient; boundary_gain is the squared crossing
   action. The assembled I + tensor metric is projected back to its analytic
   positive-semidefinite excess cone before edge integration. Boundary
   pointers may be null exactly when boundary_gain is zero.
*/
bfft_status bfft_vision_metric_edge_costs_f32(
    size_t height, size_t width,
    const float* precision_xx, const float* precision_xy,
    const float* precision_yy,
    const float* boundary_xx, const float* boundary_xy,
    const float* boundary_yy,
    double precision_gain, double boundary_gain,
    float* direction_costs);

/*
   Exact Dial-bucket first-owner walk for a direction-major float32
   eight-neighbour cost stack. delta/span/shift are the input-derived queue
   geometry; no approximate bucket width is accepted by this kernel.
*/
bfft_status bfft_vision_bucket_first_label(
    size_t height, size_t width, size_t seed_count,
    const int64_t* seed_pixel, const double* reach,
    const float* direction_costs,
    double delta, size_t span, double shift,
    int32_t* owner, double* distance, int32_t* parent,
    size_t* push_count);

/*
   Exact two-label Dial-bucket walk over the same direction-major float32
   costs. This is the owner/runner form used by nested texture transport.
   parent_first records the achieving predecessor of the winning owner.
*/
bfft_status bfft_vision_bucket_two_labels(
    size_t height, size_t width, size_t seed_count,
    const int64_t* seed_pixel, const double* reach,
    const float* direction_costs,
    double delta, size_t span, double shift,
    int32_t* owner, int32_t* runner,
    double* distance, double* second_distance,
    int32_t* parent_first, size_t* push_count);

/*
   Fit one independent conditioned affine RGB/Lab field per hard region.

   The local basis is [1, (x-cx)/r, (y-cy)/r], with normalized image
   coordinates, so the intercept is orthogonal to the two slopes. The native
   kernel fuses the repeated label reductions and uses the same closed-form
   2x2 slope solve as the reference.

   labels is HxW, target and reconstruction are HxWx3, basis is HxWx3,
   count/radius have cell_count entries, and centroid is cell_count x 2.
*/
bfft_status bfft_vision_hard_affine_fit(
    size_t height, size_t width, size_t cell_count,
    const int32_t* labels, const double* target,
    double* basis, double* count, double* radius,
    double* centroid, double* reconstruction);

/*
   Refit an already measured per-pixel hard-region basis. This is the fused
   reduction/solve/render kernel used after appending residual ridge columns.
   The first three basis columns receive the same affine regularization as
   bfft_vision_hard_affine_fit; later columns receive 2e-5 * cell mass.
*/
bfft_status bfft_vision_hard_basis_refit(
    size_t pixel_count, size_t cell_count, size_t basis_width,
    const int32_t* labels, const double* design, const double* target,
    const double* count, const double* radius, double* reconstruction);

/*
   Separable image primitives used by the V3 support lineage.  Fields are
   contiguous CxHxW float64 arrays.  The caller owns every output and scratch
   buffer, so these kernels perform no image-sized allocation.

   mirror_without_edge selects the two boundary conventions required by the
   pipeline: zero gives half-sample symmetric reflection (edge samples are
   repeated), while nonzero gives whole-sample symmetric reflection.
*/
bfft_status bfft_vision_separable_filter_f64(
    size_t channels, size_t height, size_t width,
    size_t kernel_y_size, size_t kernel_x_size,
    uint8_t mirror_without_edge, size_t thread_count,
    const double* fields, const double* kernel_y, const double* kernel_x,
    double* scratch, double* output);

/* Pixel-centred bilinear resize of a contiguous CxHxW float64 field. */
bfft_status bfft_vision_resize_bilinear_f64(
    size_t channels, size_t input_height, size_t input_width,
    size_t output_height, size_t output_width, size_t thread_count,
    const double* fields, double* output);

/* Normalized 3x3 Sobel pair for a contiguous CxHxW float64 field. */
bfft_status bfft_vision_sobel_f64(
    size_t channels, size_t height, size_t width, size_t thread_count,
    const double* fields, double* gradient_x, double* gradient_y);

/*
   Nearest-code assignment for the low-dimensional float32 palette spaces
   used by the PNG optimizer.  Ties are resolved toward the first code.
*/
bfft_status bfft_vision_nearest_code_f32(
    size_t observation_count, size_t code_count, size_t dimensions,
    size_t thread_count, const float* observations, const float* codes,
    int32_t* labels);

/* Weighted k-means++ seeding for the same palette spaces.  random_draws has
   code_count values in [0,1), allowing the Python caller to retain ownership
   of deterministic RNG state.  live_code_count receives the initialized
   prefix; a degenerate zero-distance distribution stops that prefix. */
bfft_status bfft_vision_weighted_kmeanspp_f32(
    size_t observation_count, size_t code_count, size_t dimensions,
    const float* observations, const double* weights,
    const double* random_draws, float* codes, size_t* live_code_count);

/*
   Separable orthonormal 8x8 block DCT used by the JPEG laboratory.  The
   caller supplies its 8x8 row-major transform matrix so the native and NumPy
   paths share exactly the same basis.  Forward input is HxW and output is
   ceil(H/8) x ceil(W/8) x 8 x 8; edge samples are extended.  Inverse input
   has that block shape and output is cropped directly to HxW.
*/
bfft_status bfft_vision_block_dct8_f64(
    size_t height, size_t width, size_t thread_count,
    const double* input, const double* matrix, double* coefficients);

bfft_status bfft_vision_inverse_block_dct8_f64(
    size_t height, size_t width, size_t thread_count,
    const double* coefficients, const double* matrix, double* output);

/*
   Fused RGB terminal quality metric used by the JPEG ownership optimizer.
   All image arrays are contiguous HxWxC float64 fields: RGB/reference moments
   have C=3 and reference_edge has C=2 (axis-0, axis-1 Sobel). scratch must
   contain at least 9*height*width doubles. metrics receives MSE, color SSIM,
   and Sobel MSE in that order.
*/
bfft_status bfft_vision_image_metrics_f64(
    size_t height, size_t width, size_t kernel_size, size_t thread_count,
    const double* reference, const double* candidate,
    const double* reference_mean, const double* reference_variance,
    const double* reference_edge, const double* kernel,
    double* scratch, double* metrics);

/* Repeated four-connected dilation of one HxW uint8 mask. */
bfft_status bfft_vision_binary_dilation_cross_u8(
    size_t height, size_t width, size_t iterations, size_t thread_count,
    const uint8_t* mask, uint8_t* scratch, uint8_t* output);

#ifdef __cplusplus
}
#endif

#endif
