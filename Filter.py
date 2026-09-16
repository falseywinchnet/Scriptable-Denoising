"""Cleanup — open denoising project experiment.

HEADERS / CONTRACT
==================
Edit this file, then Reload. Every function must declare an explicit Numba signature.
REQUIRED CODE STYLE: use @njit(RETURN(ARG_TYPES...), nogil=True, cache=False).
Use float64 for DSP arithmetic/storage, int64 for indices, int32 for dispositions,
and bool for predicates. VEC/MAT/CUBE below mean C-contiguous float64 arrays of
one/two/three dimensions. Use fixed storage views, and keep helper signatures
narrow; no implicit argument specialization or object mode is accepted. Python
annotations alone do NOT specify Numba compilation types. A new helper example:
    @njit(F64(VEC, I64), nogil=True, cache=False)
    def energy(samples, count):
        total = 0.0
        for i in range(count):
            total += samples[i] * samples[i]
        return total
Each helper is eagerly compiled once for its declared signature during Reload.
Reload still needs LLVM code generation; signatures do not eliminate that cost.
When a helper needs array scratch, reserve it in entity storage during Reload
and pass the view explicitly. Do not allocate temporary arrays or use NumPy's
allocating sort workspace in the audio path; use sort_values with its supplied
partition stack. Scalar-workspace helpers need no scratch-array allocation.
C++ owns STFT/ISTFT and per-channel storage. Successful audio calls go directly
to native code, without Python dispatch. Numba's generated outer error-reporting
path can still acquire the GIL; see docs/native-callback-audit.md.
A fixed native primitive implements padded recursive mask smoothing.
The loader supplies its typed address; compiled wrappers
still require explicit Numba signatures. Filter.py owns when/how they are used.
analysis/content: float64[FRAMES, BINS, 2], Cartesian real/imaginary FFT coefficients.
content is writable, analysis is read-only by contract. Frames FIRST:FIRST+EMIT are emitted; dimensions follow the headers.
state: float64[MAX_ENTITIES, ENTITY_STORAGE], initially zero on each successful reload.
config: float64[5] = sample rate, receive bandwidth Hz, squelch, reserved (formerly harmonics),
absolute sample origin of the contextual history. Script ABI 2 adds that origin.
Return PASSTHROUGH (original delayed audio), PROCESS (content), or SKIP (OLA tail,
then zeros). No FFT, I/O, variable-size persistent containers, or Python callbacks.

The initial bank ports the Downloads CleanupNative.h spectral algorithm, including
its two mask passes and three recursive smoothing passes. One bfft Hann/dual pair
replaces the old analysis-window switching. Numerical defect repairs are described
in docs/algorithm.md. Controls formerly on sliders are experiment constants here.
"""
import numpy as np
from numba import njit, types

ABI_VERSION = 2
# Deterministic channel contract: output/state depend only on the input arrays,
# this channel's fixed state, config, and immutable headers. Disable this for
# helpers with hidden mutable state (including NumPy's random generator).
# The host may reuse a channel's result while inputs and prior state coincide.
SHARE_IDENTICAL_CHANNELS = True
FFT_SIZE = 512
HOP = 128
BLOCK_SIZE = 8192
CONTEXT_BLOCKS = 3
WINDOW = "hann"  # bfft supplies the analysis window and canonical dual.
# Optional asynchronous Dear ImGui diagnostics, owned by this filter generation.
# A successful Reload launches a fresh viewer when enabled. No GUI calls run in
# compiled DSP. VIEWER_GUIDE=None displays the parent's analysis spectrum.
# A future guide entity may publish its magnitude into its own fixed state row:
# VIEWER_GUIDE = dict(slot=0, offset=..., bins=256, frames=48, first=15,
#                    emit=16, hop=512, frequency_denominator=4094, half_bin=False)
# Reserve that storage in ENTITY_STORAGE, then fill it from an explicitly typed
# helper inside Filter, e.g. publish_viewer(state, magnitude, slot, offset).
# Only the declared emitted columns are polled by the GUI.
VIEWER = True
VIEWER_AUTOSTART = True
VIEWER_GUIDE = None
EMIT = BLOCK_SIZE // HOP
FRAMES = CONTEXT_BLOCKS * EMIT
BINS = FFT_SIZE // 2 + 1
FIRST = EMIT - 1
TIME_PAD = 13
FREQ_PAD = 3
PAD_T = FRAMES + 2 * TIME_PAD
PAD_B = BINS + 2 * FREQ_PAD
PLANE = FRAMES * BINS
PAD_PLANE = PAD_T * PAD_B
PASSTHROUGH = 0
PROCESS = 1
SKIP = 2
ERROR = -1
MAX_ENTITIES = 4
ENTITY_STORAGE = 64 + 5 * PLANE + 2 * PAD_PLANE + 3 * FRAMES + 7 * BINS + max(FRAMES, PAD_T, PAD_B, 3 * BINS)
# Each entry owns a separate fixed storage row, even for repeated entity types.
# Only Cleanup is active. The retired harmonic experiment is not imported or compiled.
CHAIN = (0,)
VAD_THRESHOLD = 0.057
MASK_STRENGTH = 1.0
# Controlled mechanism ablations. Zero preserves the reference behavior.
# 1 freeze second contextual statistics; 2 binary second pass; 4 omit row gate;
# 8 omit shape scaling of contextual estimates; 16 omit residual output floor;
# 32 use row statistics alone; 64 expand guide before the same in-band Cleanup;
# 128 retain amplitudes in occupancy's second pass. Research, not UI options.
CLEANUP_AUDIT_BITS = 0
CLEANUP_AUDIT_TRACE = False
TRIANGLE = np.array([0., .14285714, .28571429, .42857143, .57142857,
                     .71428571, .85714286, 1., .85714286, .71428571,
                     .57142857, .42857143, .28571429, .14285714, 0.])

# Harmonic synthesis is retired; see research/harmonics/README.md.

# ANALYSIS EXPERIMENT — Reload reconstructs native registration plans.
# "baseline" preserves the original STFT mask; "registered" runs the COMPLETE
# Cleanup mask on the exact registered sub-hop inverse field. No Python audio calls.
ANALYSIS_MODE = "registered"
# Original Cleanup defaults to After: early evidence controls squelch; the
# mask measures the later content, in its actual post-AGC amplitude units.
# "analysis" retains the historical pre-AGC-mask experiment explicitly.
MASK_SOURCE = "content"
# Default off: keep the central enhanced-magnitude observation and skip sub-hop
# registration/fusion. Reload builds a single-lane plan; Cleanup is unchanged.
# Set True explicitly to revisit the registered experiment.
GUIDE_REGISTRATION = False
# Controlled alternative to Cleanup's MAN/ATD statistics. The listening demo
# renders both settings. Keep legacy as the reference until the trial is heard.
REGISTERED_FLOOR = "legacy"  # "legacy", "zero_crossing", or "variance_surface", "occupancy_surface"
# Bounded time/frequency moments; experimental constants, not UI controls.
# Population-only experiment: receiver (reference), oracle (test control), rolloff.
# This changes statistical membership; gains are still decided across the full guide.
CLEANUP_POPULATION = "receiver"
OCCUPANCY_SHAPE = False  # local current-distribution weighting; no quiet-frame training
SHAPE_SOFT_ADMISSION = True  # smooth relaxation of the original hard peak-seed gate
SHAPE_RADIUS_HZ = 600.0
SHAPE_STRIDE = 16  # frequency interpolation of overlapping shape neighborhoods
SURFACE_LOW_FRACTION = 0.10  # local lower-tail mean, not a noise label
SURFACE_VARIANCE_PENALTY = 1.0
SURFACE_FREQUENCY_HZ = 150.0  # half-width of local frequency moments
SURFACE_ANCHOR_SECONDS = 0.02
SURFACE_TRACK_SECONDS = 0.04
SURFACE_CLIP_MULTIPLE = 3.0
SURFACE_MEAN_RATE = 1.0
SURFACE_VARIANCE_RATE = 4.0
SURFACE_FLOOR_RATE = 1.0
# Native immutable signed analysis history, independent of the synthesis FFT.
AUDIO_HISTORY_OFFSET = 65536
AUDIO_HISTORY = dict(slot=3, offset=AUDIO_HISTORY_OFFSET) if ANALYSIS_MODE == "registered" else None
GUIDE_FFT = 2048
GUIDE_HOP = 512
GUIDE_BINS = GUIDE_FFT  # Full second-inverse nonnegative axis through Nyquist.
if REGISTERED_FLOOR in ("zero_crossing", "variance_surface", "occupancy_surface"):
    GUIDE_BINS = GUIDE_FFT  # Entire nonnegative frequency axis, including Nyquist.
if CLEANUP_AUDIT_BITS & 64 or CLEANUP_POPULATION != "receiver":
    GUIDE_BINS = GUIDE_FFT
# The second inverse divides by GUIDE_FFT-1; its magnitudes are not in the
# unnormalised FFT512 units expected by Cleanup's amplitude-dependent weights
# and second-pass residuals. Match the coherent window gain of the fixed
# reference (symmetric Hann512) to the guide (periodic Hann), including the two
# endpoint half-weights in irfft. This is a units conversion, not a gain boost
# after filtering, and does not change the registered field or its coordinates.
GUIDE_MAGNITUDE_SCALE = ((512. - 1.) * .5 * (GUIDE_FFT - 1.)) / (GUIDE_FFT * .5 - .25 * (1. - np.cos(2. * np.pi / GUIDE_FFT)))
GUIDE_FRAMES = (CONTEXT_BLOCKS * BLOCK_SIZE + GUIDE_HOP - 1) // GUIDE_HOP
GUIDE_SLOT = 2   # Native read-only input; CHAIN may not occupy this slot.
GUIDE_OFFSET = 0
REFERENCE_FRAMES = 192
REFERENCE_BINS = 257
REFERENCE_PLANE = REFERENCE_FRAMES * REFERENCE_BINS
GUIDE_FIRST = FIRST * HOP // GUIDE_HOP
GUIDE_PLANE = GUIDE_FRAMES * GUIDE_BINS
# Evaluate Cleanup on its original time lattice. The expensive registered field
# stays at GUIDE_HOP; linear interpolation supplies mask centers at 128 samples.
# This adds no observations/resolution. It preserves the original duration of
# the 15-tap triangle and recursive 13-tap time branch instead of stretching
# both fourfold when GUIDE_HOP=512. Frequency decisions remain on guide bins.
MASK_HOP = 128
MASK_FRAMES = CONTEXT_BLOCKS * BLOCK_SIZE // MASK_HOP
MASK_PLANE = MASK_FRAMES * GUIDE_BINS
GUIDE_PAD_T = MASK_FRAMES + 2 * TIME_PAD
GUIDE_PAD_B = GUIDE_BINS + 2 * FREQ_PAD
GUIDE_PAD_PLANE = GUIDE_PAD_T * GUIDE_PAD_B
GUIDE_STORAGE = 64 + 5 * MASK_PLANE + 2 * GUIDE_PAD_PLANE + 3 * max(FRAMES, MASK_FRAMES) + 7 * max(BINS, GUIDE_BINS) + max(PAD_T, GUIDE_PAD_T, GUIDE_PAD_B, 12 * GUIDE_BINS)
ENTITY_STORAGE = max(ENTITY_STORAGE, GUIDE_STORAGE, GUIDE_PLANE, 64+REFERENCE_PLANE+3*REFERENCE_FRAMES+7*REFERENCE_BINS)
ENTITY_STORAGE = max(ENTITY_STORAGE, AUDIO_HISTORY_OFFSET + CONTEXT_BLOCKS * BLOCK_SIZE)
# Surface workspace is separate from Cleanup scratch and persists across calls.
SURFACE_OFFSET = ENTITY_STORAGE
SURFACE_CHECKPOINT = SURFACE_OFFSET + 4 * MASK_PLANE
SURFACE_VIEW = None
if REGISTERED_FLOOR in ("variance_surface", "occupancy_surface"):
    ENTITY_STORAGE = SURFACE_CHECKPOINT + 3 * GUIDE_BINS + 2
    SURFACE_VIEW = dict(offset=SURFACE_OFFSET, mask_offset=64+2*MASK_PLANE,
                        frames=MASK_FRAMES, bins=GUIDE_BINS, hop=MASK_HOP)
if CLEANUP_AUDIT_BITS & 2:
    ENTITY_STORAGE = max(ENTITY_STORAGE,SURFACE_OFFSET+MASK_PLANE)
OCCUPANCY_OFFSET = ENTITY_STORAGE
if REGISTERED_FLOOR == "occupancy_surface":
    SURFACE_VIEW['low_offset'] = OCCUPANCY_OFFSET
    SURFACE_VIEW['occupancy_offset'] = OCCUPANCY_OFFSET+MASK_PLANE
    SURFACE_VIEW['reference_offset'] = OCCUPANCY_OFFSET+2*MASK_PLANE
    ENTITY_STORAGE += 3*MASK_PLANE + 2*GUIDE_BINS + 2
SHAPE_OFFSET = ENTITY_STORAGE
if OCCUPANCY_SHAPE and REGISTERED_FLOOR == "occupancy_surface":
    SURFACE_VIEW['shape_offset'] = SHAPE_OFFSET
    ENTITY_STORAGE += MASK_PLANE
# Mean |audio| -> mean absolute cosine-inverse magnitude, then the existing
# guide display units. White-noise interior-bin approximation, NOT a learned
# floor and NOT compensation for registration's maximum-selection bias.
# For C=irfft(window*x), E|C|/E|x| ~= sqrt(2*sum(window**2))/[2*(N-1)].
ZERO_CROSSING_GUIDE_SCALE = np.sqrt(2. * .375 * GUIDE_FFT) / (2. * (GUIDE_FFT-1)) * GUIDE_MAGNITUDE_SCALE
# Whole-block admission stays on the calibrated reference lattice. Guide counts
# are published as diagnostics only; no unmeasured count calibration is implied.
# Native reference analysis is always RFFT512/HOP128 over the 24576-sample context.
SORT_STACK_SIZE = 192  # 64 pending (left, right, depth) triples; float64 stores indices exactly.
SORT_STACK_OFFSET = ENTITY_STORAGE
ENTITY_STORAGE += SORT_STACK_SIZE
# Reserve scratch once per entity at Reload; pass it explicitly to every sorter.
# Sorts are sequential within an entity. Never share this writable row across channels.
if ANALYSIS_MODE == "registered":
    VIEWER_GUIDE = dict(slot=GUIDE_SLOT, offset=GUIDE_OFFSET, bins=GUIDE_BINS,
                        frames=GUIDE_FRAMES, first=(FIRST * HOP + GUIDE_HOP - 1)//GUIDE_HOP,
                        emit=BLOCK_SIZE//GUIDE_HOP, hop=GUIDE_HOP,
                        frequency_denominator=2*(GUIDE_FFT-1), half_bin=False)

# STRICT NUMBA TYPES — explicit signatures are part of the filter specification.
F64 = types.float64
I64 = types.int64
I32 = types.int32
BOOL = types.boolean
VOID = types.void
VEC = F64[::1]
MAT = F64[:, ::1]
CUBE = F64[:, :, ::1]
PAIR = types.UniTuple(F64, 2)
VAD_RESULT = types.Tuple((BOOL, F64, F64))
FILTER_SIGNATURE = I32(CUBE, CUBE, MAT, VEC)

# COMPILED METHODS
# =================

@njit(VOID(MAT, MAT, I64, I64), nogil=True, cache=False)
def publish_viewer(state, magnitude, slot, offset):
    """Optional publication from Filter into its own declared fixed storage.

    Set VIEWER_GUIDE to this slot/offset and the matrix geometry. This helper
    copies values only; the parent and asynchronous GUI own all presentation.
    """
    count = magnitude.shape[0] * magnitude.shape[1]
    if slot < 0 or slot >= state.shape[0] or offset < 0 or offset + count > state.shape[1]:
        raise ValueError('Viewer publication exceeds declared fixed storage')
    for t in range(magnitude.shape[0]):
        for b in range(magnitude.shape[1]):
            state[slot, offset + t * magnitude.shape[1] + b] = magnitude[t, b]

@njit(VOID(VEC, I64), nogil=True, cache=False)
def logistic(out, n):
    for i in range(1, n - 1):
        x = i / (n - 1.)
        out[i] = np.log(x / (1. - x))
    high = 2. * out[n - 2] - out[n - 3]
    out[0], out[n - 1] = -high, high
    for i in range(n):
        out[i] = (out[i] + high) / (2. * high)


@njit(F64(VEC, VEC, I64), nogil=True, cache=False)
def correlation(x, y, n):
    sx = sy = sxy = sxx = syy = 0.
    for i in range(n):
        sx += x[i]
        sy += y[i]
        sxy += x[i] * y[i]
        sxx += x[i] * x[i]
        syy += y[i] * y[i]
    den = (n * sxx - sx * sx) * (n * syy - sy * sy)
    if den <= 0.:
        return 1.  # Flat spectra have zero non-noise score.
    return (n * sxy - sx * sy) / np.sqrt(den)


@njit(F64(VEC, I64), nogil=True, cache=False)
def median_sorted(x, n):
    if n == 0:
        return 0.
    if n % 2:
        return x[n // 2]
    return (x[n // 2] + x[n // 2 - 1]) * .5


@njit(VOID(VEC, I64, I64), nogil=True, cache=False)
def heap_sort(x, left, right):
    """Finite-value fallback: in-place heapsort, scalar workspace only."""
    n = right - left + 1
    for start in range(n // 2 - 1, -1, -1):
        root = start
        value = x[left + root]
        while 2 * root + 1 < n:
            child = 2 * root + 1
            if child + 1 < n and x[left + child] < x[left + child + 1]:
                child += 1
            if value >= x[left + child]:
                break
            x[left + root] = x[left + child]
            root = child
        x[left + root] = value
    for end in range(n - 1, 0, -1):
        value = x[left + end]
        x[left + end] = x[left]
        root = 0
        while 2 * root + 1 < end:
            child = 2 * root + 1
            if child + 1 < end and x[left + child] < x[left + child + 1]:
                child += 1
            if value >= x[left + child]:
                break
            x[left + root] = x[left + child]
            root = child
        x[left + root] = value


@njit(VOID(VEC, I64, VEC), nogil=True, cache=False)
def sort_values(x, n, stack):
    """Finite-value introsort with a caller-owned partition stack.

    Process the smaller partition first so pending partitions use O(log n)
    triples. Depth exhaustion switches to scalar-workspace heapsort. Neither
    path allocates. The input and stack must be distinct fixed storage views.
    """
    if n < 2:
        return
    left, right, depth, size, used = 0, n - 1, 0, n, 0
    while size > 1:
        depth += 2
        size //= 2
    while True:
        while right - left >= 16:
            if depth == 0:
                heap_sort(x, left, right)
                left = right
                break
            depth -= 1
            a, b, c = x[left], x[(left + right) // 2], x[right]
            if a > b: a, b = b, a
            if b > c: b, c = c, b
            if a > b: a, b = b, a
            pivot = b
            i, j = left, right
            while i <= j:
                while x[i] < pivot: i += 1
                while x[j] > pivot: j -= 1
                if i <= j:
                    x[i], x[j] = x[j], x[i]
                    i += 1
                    j -= 1
            if j - left < right - i:
                pending_left, pending_right = i, right
                right = j
            else:
                pending_left, pending_right = left, j
                left = i
            if pending_left < pending_right:
                if used + 3 > stack.size:
                    raise ValueError('sort partition scratch is too small')
                stack[used], stack[used + 1], stack[used + 2] = pending_left, pending_right, depth
                used += 3
        for i in range(left + 1, right + 1):
            value, j = x[i], i - 1
            while j >= left and x[j] > value:
                x[j + 1] = x[j]
                j -= 1
            x[j + 1] = value
        if used == 0:
            return
        used -= 3
        left, right, depth = int(stack[used]), int(stack[used + 1]), int(stack[used + 2])


@njit(F64(VEC, I64), nogil=True, cache=False)
def median_select(x, n):
    """Exact finite-value median in caller storage; no complete ordering needed.

    Partition to the upper middle element; the maximum of its lower partition
    supplies the other middle element for even n. A depth limit retains the
    scalar-workspace heapsort as a fallback for adversarial partitions.
    """
    if n == 0:
        return 0.
    k = n // 2
    left, right = 0, n - 1
    budget, size = 2, n
    while size > 1:
        budget += 2
        size //= 2
    while left < right:
        if budget == 0:
            heap_sort(x, left, right)
            break
        budget -= 1
        a, b, c = x[left], x[(left+right)//2], x[right]
        if a > b: a, b = b, a
        if b > c: b, c = c, b
        if a > b: a, b = b, a
        pivot = b
        i, j = left, right
        while i <= j:
            while x[i] < pivot: i += 1
            while x[j] > pivot: j -= 1
            if i <= j:
                x[i], x[j] = x[j], x[i]
                i += 1
                j -= 1
        if k <= j:
            right = j
        elif k >= i:
            left = i
        else:
            break
    upper = x[k]
    if n % 2:
        return upper
    lower = x[0]
    for i in range(1, k):
        lower = max(lower, x[i])
    return (upper + lower) * .5


@njit(types.UniTuple(F64, 3)(MAT, I64, I64), nogil=True, cache=False)
def support_line(prefix, lo, hi):
    """OLS on a segment of a robust log-envelope; no signal/noise labels."""
    n=float(hi-lo)
    sx=prefix[hi,0]-prefix[lo,0];sy=prefix[hi,1]-prefix[lo,1]
    xx=prefix[hi,2]-prefix[lo,2];xy=prefix[hi,3]-prefix[lo,3]
    yy=prefix[hi,4]-prefix[lo,4]
    den=n*xx-sx*sx
    slope=(n*xy-sx*sy)/den if den>1e-20 else 0.
    intercept=(sy-slope*sx)/n
    error=max(0.,yy-intercept*sy-slope*xy)
    return intercept,slope,error


@njit(types.UniTuple(F64, 3)(MAT, VEC, VEC, I64, I64), nogil=True, cache=False)
def support_robust_line(prefix, x, y, lo, hi):
    """Huber line fit: bounded influence from a carrier or an internal notch."""
    a,b,_=support_line(prefix,lo,hi)
    delta=.35  # natural-log amplitude (~3 dB), envelope-model residual scale
    for iteration in range(12):
        sw=sx=sy=xx=xy=0.
        for j in range(lo,hi):
            residual=abs(y[j]-a-b*x[j])
            w=min(1.,delta/max(residual,1e-20))
            sw+=w;sx+=w*x[j];sy+=w*y[j];xx+=w*x[j]*x[j];xy+=w*x[j]*y[j]
        den=sw*xx-sx*sx
        if den<=1e-20:break
        b=(sw*xy-sx*sy)/den;a=(sy-b*sx)/sw
    loss=0.
    for j in range(lo,hi):
        r=abs(y[j]-a-b*x[j])
        loss+=r*r if r<=delta else 2.*delta*r-delta*delta
    return a,b,loss


@njit(I64(MAT, F64, F64, VEC, MAT, VEC, VEC, VEC), nogil=True, cache=False)
def population_support(data, rate, denominator, envelope, prefix, scratch, report, sort_stack):
    """Experimental LPF-population knee, never a temporal noise classifier.

    Aggregate ALL time columns, take local frequency medians, then fit a broad
    descending log-frequency segment after a free-slope prefix, with optional
    local windows. The outermost distinct knee estimates the population edge.
    Full support is the fallback. Constants describe this rolloff hypothesis;
    this is not a calibrated probability or a universal passband detector.
    Caller owns bins + 96*2 scratch, 97x5 prefix, and 8 report values.
    """
    rows,bins=data.shape
    df=rate/denominator
    for b in range(bins):
        total=0.
        for t in range(rows):total+=max(0.,data[t,b])
        envelope[b]=total/rows
    report[:]=0.
    maximum=np.max(envelope)
    if maximum<=0.:return bins
    count=96
    x=scratch[bins:bins+count];y=scratch[bins+count:bins+2*count]
    lower=max(4.*df,rate/256.)
    upper=(bins-1)*df
    if upper<=lower*2. or bins<576:return bins
    for j in range(count):
        f=lower*np.exp(np.log(upper/lower)*j/(count-1))
        center=int(np.floor(f/df+.5))
        radius=max(2,int(.06*f/df))
        lo=max(0,center-radius);hi=min(bins,center+radius+1)
        n=hi-lo
        for b in range(n):scratch[b]=envelope[lo+b]
        sort_values(scratch[:n], (scratch[:n]).size, sort_stack)
        value=median_sorted(scratch,n)
        x[j]=np.log(f)
        y[j]=np.log(max(value,maximum*1e-14))
    prefix[:]=0.
    for j in range(count):
        prefix[j+1,0]=prefix[j,0]+x[j]
        prefix[j+1,1]=prefix[j,1]+y[j]
        prefix[j+1,2]=prefix[j,2]+x[j]*x[j]
        prefix[j+1,3]=prefix[j,3]+x[j]*y[j]
        prefix[j+1,4]=prefix[j,4]+y[j]*y[j]
    # Search local slope changes rather than letting a strong low-frequency
    # feature determine the shape of the entire prefix. An internal valley can
    # contribute a candidate; choose the outermost separated knee cluster.
    best=-1.;edge=upper;best_slope=0.;best_prefix=0.;null_error=best_error=0.
    candidates=scratch[:576].reshape((6,96));candidates[:]=0.
    for j in range(8,count-14):
        lo=j-8;hi=j+14
        a,b,e1=support_robust_line(prefix,x,y,lo,j)
        c,d,e2=support_robust_line(prefix,x,y,j,hi)
        # A steep recovery from an internal notch is not an outer LPF knee.
        if b>2. or d>=-2. or b-d<2.:continue
        crossing=(c-a)/(b-d)
        if crossing<x[j-3] or crossing>x[j+3]:continue
        _,_,e0=support_robust_line(prefix,x,y,lo,hi)
        improvement=(e0-e1-e2)/(.35*.35)-3.*np.log(float(hi-lo))
        if improvement<=10.:continue
        candidates[0,j]=improvement;candidates[1,j]=np.exp(crossing)
        candidates[2,j]=b;candidates[3,j]=d;candidates[4,j]=e0;candidates[5,j]=e1+e2
    # Distinct local maxima, not connected threshold crossings: a notch's
    # recovery can otherwise merge with the later receiver rolloff candidate.
    for j in range(8,count-14):
        score=candidates[0,j]
        if score<=10.:continue
        if score<np.max(candidates[0,j-3:j+4]):continue
        best=score;edge=candidates[1,j];best_prefix=candidates[2,j]
        best_slope=candidates[3,j];null_error=candidates[4,j];best_error=candidates[5,j]
    report[0]=edge;report[1]=max(0.,best);report[2]=best_prefix
    report[3]=best_slope;report[4]=null_error;report[5]=best_error
    report[6]=1. if best>10. else 0.
    if best<=10.:return bins
    return min(bins,max(4,int(np.floor(edge/df+1.5))))


@njit(PAIR(MAT, I64, VEC, VEC), nogil=True, cache=False)
def statistics(data, nbins, values, deviations):
    n = 0
    for t in range(data.shape[0]):
        for b in range(nbins):
            v = data[t, b]
            if v != 0. and np.isfinite(v):
                values[n] = v
                n += 1
    if n == 0:
        return 0., 0.
    med = median_select(values, n)
    for i in range(n):
        deviations[i] = abs(values[i] - med)
    man = median_select(deviations, n)
    ss = 0.
    for t in range(data.shape[0]):
        for b in range(nbins):
            ss += (data[t, b] - man) ** 2
    return man, np.sqrt(ss / (data.shape[0] * nbins)) - man


@njit(PAIR(VEC, I64, VEC, VEC), nogil=True, cache=False)
def row_statistics(data, nbins, values, deviations):
    n = 0
    for b in range(nbins):
        if data[b] != 0.:
            values[n] = data[b]
            n += 1
    if n == 0:
        return 0., 0.
    med = median_select(values, n)
    # Preserve the original first-n deviations convention, including zeros.
    for b in range(n):
        deviations[b] = abs(data[b] - med)
    man = median_select(deviations, n)
    ss = 0.
    for b in range(nbins):
        ss += (data[b] - man) ** 2
    return man, np.sqrt(ss / nbins)


@njit(VOID(VEC, F64, I64, F64), nogil=True, cache=False)
def remove_runs(a, value, threshold, replacement):
    first = 0
    while first < a.size:
        end = first + 1
        while end < a.size and a[end] == a[first]:
            end += 1
        # Original uses run length + 1; retained for the stage-one baseline.
        if a[first] == value and end - first + 1 < threshold:
            for i in range(first, end):
                a[i] = replacement
        first = end


@njit(VAD_RESULT(MAT, I64, VEC, VEC, VEC, VEC, VEC, VEC, VEC), nogil=True, cache=False)
def entropy(data, nb, raw, smooth, detected, log1, log3, scratch, sort_stack):
    logistic(log1, nb)
    logistic(log3, nb * 3)
    scratch[:nb * 3] = 0.
    scratch[nb - 1] = 1.
    maximum = 1. - correlation(scratch, log1, nb)
    scratch[nb - 1] = 0.
    scratch[nb * 3 - 1] = 1.
    maximum3 = 1. - correlation(scratch, log3, nb * 3)
    for t in range(data.shape[0]):
        lo, hi = t - 1, t + 2
        if t == 0 or t == data.shape[0] - 1:
            lo, hi = t, t + 1
        n = 0
        for j in range(lo, hi):
            for b in range(nb):
                scratch[n] = data[j, b]
                n += 1
        sort_values(scratch[:n], (scratch[:n]).size, sort_stack)
        # Pearson correlation is affine invariant; no in-place normalization.
        if n == nb:
            raw[t] = 1. - correlation(scratch, log1, n)
        else:
            raw[t] = 1. - correlation(scratch, log3, n)
    count = streak = longest = 0
    for t in range(data.shape[0]):
        v = raw[t]
        if t > 0:
            v += raw[t - 1]
        if t < data.shape[0] - 1:
            v += raw[t + 1]
        smooth[t] = v / 3.
        detected[t] = 1. if smooth[t] > VAD_THRESHOLD else 0.
        if detected[t] != 0.:
            streak += 1
            longest = max(longest, streak)
            if EMIT // 2 - 1 < t < FRAMES - EMIT // 2 + 1:
                count += 1
        else:
            streak = 0
    active = count > int(22 * 128 / HOP) or longest > int(16 * 128 / HOP)
    if active:
        remove_runs(detected, 0., 6, 1.)
        remove_runs(detected, 1., 2, 0.)
    return active, maximum, maximum3


@njit(VAD_RESULT(MAT, I64, VEC, VEC, VEC, VEC, VEC, VEC, VEC), nogil=True, cache=False)
def reference_entropy(data, nb, raw, smooth, detected, log1, log3, scratch, sort_stack):
    logistic(log1, nb)
    logistic(log3, nb * 3)
    scratch[:nb * 3] = 0.
    scratch[nb - 1] = 1.
    maximum = 1. - correlation(scratch, log1, nb)
    scratch[nb - 1] = 0.
    scratch[nb * 3 - 1] = 1.
    maximum3 = 1. - correlation(scratch, log3, nb * 3)
    for t in range(data.shape[0]):
        lo, hi = t - 1, t + 2
        if t == 0 or t == data.shape[0] - 1:
            lo, hi = t, t + 1
        n = 0
        for j in range(lo, hi):
            for b in range(nb):
                scratch[n] = data[j, b]
                n += 1
        sort_values(scratch[:n], (scratch[:n]).size, sort_stack)
        # Pearson correlation is affine invariant; no in-place normalization.
        if n == nb:
            raw[t] = 1. - correlation(scratch, log1, n)
        else:
            raw[t] = 1. - correlation(scratch, log3, n)
    count = streak = longest = 0
    for t in range(data.shape[0]):
        v = raw[t]
        if t > 0:
            v += raw[t - 1]
        if t < data.shape[0] - 1:
            v += raw[t + 1]
        smooth[t] = v / 3.
        detected[t] = 1. if smooth[t] > VAD_THRESHOLD else 0.
        if detected[t] != 0.:
            streak += 1
            longest = max(longest, streak)
            if 32 <= t < 161:
                count += 1
        else:
            streak = 0
    active = count > 22 or longest > 16
    if active:
        remove_runs(detected, 0., 6, 1.)
        remove_runs(detected, 1., 2, 0.)
    return active, maximum, maximum3


@njit(VAD_RESULT(MAT, I64, VEC, VEC, VEC, VEC, VEC, VEC, BOOL, VEC), nogil=True, cache=False)
def guide_entropy(data, nb, raw, smooth, detected, log1, log3, scratch, active, sort_stack):
    logistic(log1, nb)
    logistic(log3, nb * 3)
    scratch[:nb * 3] = 0.
    scratch[nb - 1] = 1.
    maximum = 1. - correlation(scratch, log1, nb)
    scratch[nb - 1] = 0.
    scratch[nb * 3 - 1] = 1.
    maximum3 = 1. - correlation(scratch, log3, nb * 3)
    for t in range(data.shape[0]):
        lo, hi = t - 1, t + 2
        if t == 0 or t == data.shape[0] - 1:
            lo, hi = t, t + 1
        n = 0
        for j in range(lo, hi):
            for b in range(nb):
                scratch[n] = data[j, b]
                n += 1
        sort_values(scratch[:n], (scratch[:n]).size, sort_stack)
        # Pearson correlation is affine invariant; no in-place normalization.
        if n == nb:
            raw[t] = 1. - correlation(scratch, log1, n)
        else:
            raw[t] = 1. - correlation(scratch, log3, n)
    for t in range(data.shape[0]):
        v = raw[t]
        if t > 0:
            v += raw[t - 1]
        if t < data.shape[0] - 1:
            v += raw[t + 1]
        smooth[t] = v / 3.
        detected[t] = 1. if smooth[t] > VAD_THRESHOLD else 0.
    if active:
        remove_runs(detected, 0., 6, 1.)
        remove_runs(detected, 1., 2, 0.)
    return active, maximum, maximum3


@njit(VOID(MAT, MAT, I64, VEC), nogil=True, cache=False)
def sawtooth(data, out, nb, scratch):
    # Adjacent bins share cache lines. Buffer complete time columns in small
    # tiles before publishing: out may alias data, including on the third pass.
    # Keep the original tap order and literal divisor for each output cell.
    rows = data.shape[0]
    if rows == 0 or nb == 0:
        return
    tile = min(32, scratch.size // rows)
    if tile == 0:
        raise ValueError('sawtooth needs at least one time column of scratch')
    for first in range(0, nb, tile):
        width = min(tile, nb - first)
        for t in range(data.shape[0]):
            for b in range(width):
                scratch[t * width + b] = 0.
            for k in range(15):
                j = t + k - 7
                if 0 <= j < rows:
                    for b in range(width):
                        scratch[t * width + b] += data[j, first + b] * TRIANGLE[k]
        for t in range(data.shape[0]):
            for b in range(width):
                out[t, first + b] = scratch[t * width + b] / 6.916666666666666666


@njit(PAIR(MAT, MAT, I64, VEC, VEC, F64, BOOL, F64, F64, VEC, VEC, I64), nogil=True, cache=False)
def population_peaks(data, mask, nb, raw, detected, maximum, squelch, man, atd, values, deviations, population_bins):
    threshold_sum = accepted = rows = 0.
    for t in range(data.shape[0]):
        if detected[t] == 0. and not (CLEANUP_AUDIT_BITS & 4):
            if squelch or raw[t] < VAD_THRESHOLD:
                continue
        rm, ra = row_statistics(data[t], population_bins, values, deviations)
        f = 1. if CLEANUP_AUDIT_BITS & 8 else 1. - raw[t] / maximum
        gm, ga = man * f, atd * f
        wm, wa = np.exp(-.5 * abs(gm - rm)), np.exp(-.5 * abs(ga - ra))
        threshold = rm * wm + gm * (1. - wm) + MASK_STRENGTH * (ra * wa + ga * (1. - wa))
        if CLEANUP_AUDIT_BITS & 32:threshold=rm+MASK_STRENGTH*ra
        threshold_sum += threshold
        rows += 1.
        # Read the complete row before writing: second pass intentionally aliases.
        for b in range(nb):
            if data[t, b] > threshold:
                mask[t, b] = 1.
                accepted += 1.
    return threshold_sum / max(rows, 1.), accepted / (data.shape[0] * nb)


@njit(PAIR(MAT, MAT, I64, VEC, VEC, F64, BOOL, F64, F64, VEC, VEC), nogil=True, cache=False)
def peaks(data, mask, nb, raw, detected, maximum, squelch, man, atd, values, deviations):
    return population_peaks(data,mask,nb,raw,detected,maximum,squelch,man,atd,values,deviations,nb)


@njit(I32(MAT, MAT, MAT, I64, VEC), nogil=True, cache=False)
def recursive_smooth(mask, vertical, horizontal, nb, scratch):
    """Exact padded two-branch recurrence, evaluated by the native primitive.

    Frequency and time branches are averaged after each round. This is not a
    sequential separable blur. The parent supplies typed native function pointers
    at Reload; the call has no Python dispatch, allocation, or extra worker.
    """
    rows, columns = mask.shape
    padded_rows, padded_columns = rows + 2 * TIME_PAD, columns + 2 * FREQ_PAD
    if (vertical.shape[0] != padded_rows or vertical.shape[1] != padded_columns or
            horizontal.shape[0] != padded_rows or horizontal.shape[1] != padded_columns or
            scratch.size < max(padded_rows, padded_columns)):
        return -1
    return _recursive_smooth(mask.ctypes, vertical.ctypes, horizontal.ctypes,
                             scratch.ctypes, rows, columns, nb, TIME_PAD, FREQ_PAD, 3)


@njit(I32(CUBE, CUBE, VEC, VEC), nogil=True, cache=False)
def cleanup(analysis, content, state, config):
    sort_stack = state[SORT_STACK_OFFSET:SORT_STACK_OFFSET + SORT_STACK_SIZE]
    nb = min(BINS, max(4, int(np.floor(config[1] / (config[0] / FFT_SIZE) + 1.5))))
    squelch = config[2] != 0.
    # All working arrays are views of this entity's persistent fixed storage.
    offset = 64
    mag = state[offset:offset + PLANE].reshape((FRAMES, BINS)); offset += PLANE
    smoothed = state[offset:offset + PLANE].reshape((FRAMES, BINS)); offset += PLANE
    mask = state[offset:offset + PLANE].reshape((FRAMES, BINS)); offset += PLANE
    vertical = state[offset:offset + PAD_PLANE].reshape((PAD_T, PAD_B)); offset += PAD_PLANE
    horizontal = state[offset:offset + PAD_PLANE].reshape((PAD_T, PAD_B)); offset += PAD_PLANE
    values = state[offset:offset + PLANE]; offset += PLANE
    deviations = state[offset:offset + PLANE]; offset += PLANE
    raw = state[offset:offset + FRAMES]; offset += FRAMES
    smooth = state[offset:offset + FRAMES]; offset += FRAMES
    detected = state[offset:offset + FRAMES]; offset += FRAMES
    log1 = state[offset:offset + BINS]; offset += BINS
    log3 = state[offset:offset + 3 * BINS]; offset += 3 * BINS
    scratch = state[offset:]
    mag[:] = 0.
    smoothed[:] = 0.
    mask[:] = 0.
    for t in range(FRAMES):
        for b in range(nb):
            mag[t, b] = np.hypot(analysis[t, b, 0], analysis[t, b, 1])
    active, maximum, maximum3 = entropy(mag, nb, raw, smooth, detected, log1, log3, scratch, sort_stack)
    state[16:24] = 0.  # Per-block floor/threshold diagnostics.
    state[0] += 1.  # Persistent segment counter, useful to other entities/tests.
    state[1] = 1. if active else 0.
    if squelch and not active:
        return SKIP
    if MASK_SOURCE == "content":
        for t in range(FRAMES):
            for b in range(nb):
                mag[t,b] = np.hypot(content[t,b,0], content[t,b,1])
    initial = np.max(mag)
    if initial <= 0.:
        return SKIP if squelch else PASSTHROUGH
    sawtooth(mag, smoothed, nb, scratch)
    man, atd = statistics(smoothed, nb, values, deviations)
    state[16], state[17] = man, atd
    state[18], state[19] = peaks(smoothed, mask, nb, raw, detected, maximum, squelch, man, atd, values, deviations)
    for t in range(FRAMES):
        for b in range(nb):
            if mask[t, b] == 0.:
                mag[t, b] = 0.
    multiplier = min(1., np.max(mag) / initial)
    man, atd = statistics(mag, nb, values, deviations)
    sawtooth(mag, smoothed, nb, scratch)
    state[20], state[21] = man, atd
    state[22], state[23] = peaks(smoothed, smoothed, nb, raw, detected, maximum, squelch, man, atd, values, deviations)
    for t in range(FRAMES):
        for b in range(nb):
            mask[t, b] = min(mask[t, b], smoothed[t, b] * multiplier)
    sawtooth(mask, mask, nb, scratch)
    if recursive_smooth(mask, vertical, horizontal, nb, scratch) != 0:
        return ERROR
    if not squelch:
        for t in range(FRAMES):
            for b in range(nb):
                source = content if MASK_SOURCE == "content" else analysis
                mag[t, b] = np.hypot(source[t, b, 0], source[t, b, 1])
        man, atd = statistics(mag, nb, values, deviations)
        # Legacy extrema started at zero. Fix stale extrema across calls.
        maximum_value = np.max(mag)
        if maximum_value > 0.:
            man /= maximum_value
            atd /= maximum_value
    for t in range(FIRST, FIRST + EMIT):
        for b in range(BINS):
            gain = mask[t, b] if b < nb else 0.
            if not squelch and b < nb and detected[t - FIRST] == 0.:
                # Preserve legacy output-row indexing pending stage-two VAD work.
                gain = max(gain, man * smooth[t] / maximum3 + atd * (1. - MASK_STRENGTH))
            content[t, b, 0] *= gain
            content[t, b, 1] *= gain
    return PROCESS


@njit(VOID(VEC, VEC, I64, I64), nogil=True, cache=False)
def record_evidence(detected, state, offset, hop):
    """Count the reference region's center coordinates and full-context run.

    Native guide density changes available intervals; these values are diagnostics,
    NOT a new calibrated admission rule. The old [32,161) region is retained in
    sample coordinates at FFT512/HOP128. Slots offset: count, available, run, total.
    """
    count = available = streak = longest = 0
    for t in range(detected.size):
        inside = 32 * 128 <= t * hop < 161 * 128
        if inside:
            available += 1
        if detected[t] > VAD_THRESHOLD:
            streak += 1
            longest = max(longest, streak)
            if inside:
                count += 1
        else:
            streak = 0
    state[offset] = count
    state[offset+1] = available
    state[offset+2] = longest
    state[offset+3] = detected.size


@njit(VOID(MAT, CUBE, I64, F64), nogil=True, cache=False)
def apply_guide_gain(gain, content, nb, half_bin):
    """Bilinear real gain at actual bin/frame centers; phase stays in content.

    Guide index r has frequency r*fs/[2*(GUIDE_FFT-1)]. Outside the declared
    receive band gain is zero (legacy behavior). An uncovered guide coordinate
    is neutral; cleanup_registered rejects inadequate receive-band coverage.
    """
    for t in range(FIRST, FIRST + EMIT):
        x = t * HOP / MASK_HOP
        t0 = int(np.floor(x))
        tf = x - t0
        for b in range(BINS):
            y = (b + half_bin) * (2. * (GUIDE_FFT - 1)) / FFT_SIZE
            b0 = int(np.floor(y))
            bf = y - b0
            value = 1.
            if b >= nb:
                value = 0.
            elif 0. <= x <= gain.shape[0]-1 and 0. <= y <= gain.shape[1]-1:
                t1 = min(t0 + 1, gain.shape[0] - 1)
                b1 = min(b0 + 1, gain.shape[1] - 1)
                value = ((1.-bf)*gain[t0,b0]+bf*gain[t0,b1])*(1.-tf) + ((1.-bf)*gain[t1,b0]+bf*gain[t1,b1])*tf
                value = min(1., max(0., value))
            content[t,b,0] *= value
            content[t,b,1] *= value


@njit(PAIR(VEC), nogil=True, cache=False)
def crossing_neighbors(samples):
    """Mean absolute bounding samples, two observations per sign crossing.

    Skip exact-zero runs; count their nonzero endpoints only when signs differ.
    A sample bounding two crossings is counted twice (each event has two sides).
    No centering, clipping, quantiles, or amplitude-dependent selection. With no
    crossings return (0, 0); zero means unavailable, not evidence of no noise.
    """
    previous = 0.
    total = count = 0.
    for value in samples:
        if value == 0.:
            continue
        if (value < 0. and previous > 0.) or (value > 0. and previous < 0.):
            total += abs(previous) + abs(value)
            count += 2.
        previous = value
    return total / max(count, 1.), count


@njit(PAIR(MAT, MAT, I64, F64), nogil=True, cache=False)
def crossing_peaks(data, mask, nb, floor):
    # Same binary seeding and second-pass residual convention as Cleanup,
    # but no MAN/ATD, row fit, entropy weighting, or relative-peak adjustment.
    accepted = 0.
    for t in range(data.shape[0]):
        for b in range(nb):
            if data[t,b] > floor:
                mask[t,b] = 1.
                accepted += 1.
    return floor, accepted / (data.shape[0] * nb)


@njit(I32(MAT, VEC, MAT, VEC, F64, I64, I64, VEC), nogil=True, cache=False)
def research_floor_ratio(data, floor, ratio, scratch, quantile, first, stop, sort_stack):
    """Research-only self-normalization; NOT connected to FilterBank.

    One temporal quantile per frequency, before time interpolation. No pooled
    frequency distribution and no receive-bandwidth argument. The quantile is
    a scale statistic, not a claimed mean noise amplitude. A noise-only run of
    this exact procedure must calibrate its ratio distribution and threshold.
    Zero scale means insufficient background evidence; ratio=0 is an abstention
    in this research output, NOT authorization to suppress live audio.
    """
    rows, bins = data.shape
    count = stop - first
    if (first < 0 or stop > rows or count < 2 or quantile < 0. or quantile > 1.
            or floor.size != bins or ratio.shape != data.shape or scratch.size < count):
        return -1
    at = quantile * (count-1)
    low = int(at)
    high = min(low+1, count-1)
    fraction = at-low
    for b in range(bins):
        for t in range(count):
            scratch[t] = data[first+t,b]
        sort_values(scratch[:count], (scratch[:count]).size, sort_stack)
        scale = scratch[low]*(1.-fraction) + scratch[high]*fraction
        floor[b] = scale
        for t in range(rows):
            ratio[t,b] = data[t,b]/scale if scale > 1e-280 else 0.
    return 0


@njit(VOID(MAT, MAT, F64), nogil=True, cache=False)
def research_tail_seeds(score, seeds, threshold):
    """Calibrated upper-tail event seeds, not a finished denoising gain."""
    for t in range(score.shape[0]):
        for b in range(score.shape[1]):
            seeds[t,b] = 1. if score[t,b] > threshold else 0.


@njit(I32(VEC, VEC, VEC, I64, I64, VEC), nogil=True, cache=False)
def research_guarded_floor(temporal, joint, scratch, guard, width, sort_stack):
    """Research-only second reference for persistent, spectrally narrow ridges.

    Exclude the tested ridge and use the GREATER median of the two flanks.
    Thus one suppressed flank at an LPF edge cannot lower the reference alone.
    The minimum with the temporal estimate is a union of two tests; calibrate
    that union on noise, not each component independently at the same rate.
    Requires smooth background over the reference span; not a universal prior.
    Missing/zero reference cells cause abstention from spectral correction.
    """
    if guard < 0 or width < 2 or scratch.size < width or joint.size != temporal.size:
        return -1
    for b in range(temporal.size):
        joint[b] = temporal[b]
        if b-guard-width < 0 or b+guard+width >= temporal.size:
            continue
        background = 0.
        valid = True
        for side in range(2):
            for j in range(width):
                k = b-guard-width+j if side == 0 else b+guard+1+j
                scratch[j] = temporal[k]
                if scratch[j] <= 1e-280:
                    valid = False
            sort_values(scratch[:width], (scratch[:width]).size, sort_stack)
            background = max(background, median_sorted(scratch, width))
        if valid:
            joint[b] = min(temporal[b], background)
    return 0


@njit(I32(MAT, VEC, VEC, MAT, VEC, I64, I64, VEC), nogil=True, cache=False)
def research_joint_ratio(data, temporal, joint, ratio, scratch, guard, width, sort_stack):
    """Apply the guarded reference; retain all candidate DSP in this file."""
    status = research_guarded_floor(temporal, joint, scratch, guard, width, sort_stack)
    if status != 0:
        return status
    for t in range(data.shape[0]):
        for b in range(data.shape[1]):
            ratio[t,b] = data[t,b]/joint[b] if joint[b] > 1e-280 else 0.
    return 0


@njit(I32(MAT, MAT, MAT, MAT, MAT, VEC), nogil=True, cache=False)
def research_pool_score(ratio, output, dense, vertical, horizontal, scratch):
    """Test evidence pooling with Cleanup's existing padded recurrence.

    Interpolation creates no new statistical observations. Calibrate the final
    pooled score separately. This experiment is not the live mask path.
    """
    factor = dense.shape[0] // ratio.shape[0]
    for t in range(dense.shape[0]):
        at = t / float(factor)
        low = int(at)
        high = min(low+1, ratio.shape[0]-1)
        fraction = at-low
        for b in range(ratio.shape[1]):
            dense[t,b] = ratio[low,b]*(1.-fraction) + ratio[high,b]*fraction
    status = recursive_smooth(dense, vertical, horizontal, ratio.shape[1], scratch)
    if status != 0:
        return status
    for t in range(output.shape[0]):
        for b in range(output.shape[1]):
            output[t,b] = dense[t*factor,b]
    return 0


@njit(VOID(MAT, MAT, MAT, MAT, MAT, MAT, VEC, VEC, F64, F64, VEC), nogil=True, cache=False)
def variance_surface(data, means, variances, floors, clipped, checkpoint, meta, scratch, sample_rate, origin, sort_stack):
    """Bounded local time/frequency moments; no source-presence classification.

    Nine frequency/forward-time anchors bound each observation's contribution. The
    original data remain untouched. Frequency moments use a sliding window;
    temporal moments and the penalized floor use bounded innovations. Store a
    checkpoint ONE block into this context, so overlapping contexts never train
    the tracker repeatedly. The remaining context is lookahead for this call.
    Initial positive observation seeds its local scale; zero has no relative
    scale from which to bound a first observation. Reload resets all checkpoints.
    """
    rows, bins = data.shape
    radius = max(1, int(SURFACE_FREQUENCY_HZ * 2.*(GUIDE_FFT-1)/sample_rate))
    tr = max(1, int(SURFACE_ANCHOR_SECONDS*sample_rate/MASK_HOP))
    # Bounded-influence observations: median/MAD from a 3x3 anchor stencil.
    for t in range(rows):
        for b in range(bins):
            k = 0
            for dt in range(-1,2):
                for df in range(-1,2):
                    scratch[k] = data[min(rows-1,max(0,t+(dt+1)*tr)),min(bins-1,max(0,b+df*radius))]
                    k += 1
            sort_values(scratch[:9], (scratch[:9]).size, sort_stack)
            center = scratch[4]
            for k in range(9):
                scratch[k] = abs(scratch[k]-center)
            sort_values(scratch[:9], (scratch[:9]).size, sort_stack)
            cap = center + SURFACE_CLIP_MULTIPLE*max(center,scratch[4])
            clipped[t,b] = min(max(0.,data[t,b]),cap)
    # Sliding frequency moments, O(rows*bins), with clipped observations.
    for t in range(rows):
        total = total2 = 0.
        left = 0
        right = min(bins,radius+1)
        for b in range(right):
            x = clipped[t,b]; total += x; total2 += x*x
        for b in range(bins):
            count = right-left
            mu = max(0.,total/count)
            means[t,b] = mu
            variances[t,b] = max(0.,total2/count-mu*mu)
            next_left = max(0,b+1-radius)
            next_right = min(bins,b+radius+2)
            if next_left > left:
                x = clipped[t,left]; total -= x; total2 -= x*x
            if next_right > right:
                x = clipped[t,right]; total += x; total2 += x*x
            left,right = next_left,next_right
    alpha = 1.-np.exp(-MASK_HOP/(sample_rate*SURFACE_TRACK_SECONDS))
    continuous = meta[0] != 0. and origin == meta[1]+BLOCK_SIZE
    step = BLOCK_SIZE//MASK_HOP
    for b in range(bins):
        mu = checkpoint[0,b] if continuous else 0.
        var = checkpoint[1,b] if continuous else 0.
        floor = checkpoint[2,b] if continuous else 0.
        for t in range(rows):
            local_mu,local_var = means[t,b],variances[t,b]
            if mu <= 0.:
                mu,var = local_mu,local_var
                floor = mu/(1.+SURFACE_VARIANCE_PENALTY*(var/mu)/mu) if mu > 0. else 0.
            else:
                scale = max(mu,np.sqrt(var))
                delta = local_mu-mu
                bounded = min(SURFACE_MEAN_RATE*scale,max(-SURFACE_MEAN_RATE*scale,delta))
                # Winsorize the innovation before squaring as well.
                target_var = local_var+(1.-alpha)*bounded*bounded
                limit = SURFACE_VARIANCE_RATE*scale*scale
                var = max(0.,var+alpha*min(limit,max(-limit,target_var-var)))
                mu = max(0.,mu+alpha*bounded)
                target = mu/(1.+SURFACE_VARIANCE_PENALTY*(var/mu)/mu) if mu > 0. else 0.
                limit_floor = alpha*SURFACE_FLOOR_RATE*scale
                floor = max(0.,floor+min(limit_floor,max(-limit_floor,target-floor)))
            means[t,b],variances[t,b],floors[t,b] = mu,var,floor
            if t == step-1:
                checkpoint[0,b],checkpoint[1,b],checkpoint[2,b] = mu,var,floor
    meta[0],meta[1] = 1.,origin


@njit(VOID(MAT, MAT, MAT, MAT, MAT, MAT, MAT, VEC, VEC, VEC, F64, F64, VEC), nogil=True, cache=False)
def occupancy_surface(clipped, means, variances, floor, low, occupancy, reference, checkpoint, meta, scratch, rate, origin, sort_stack):
    """Low-tail anchor + RMS excursion: baseline-shaped decision contour.

    L is the local bottom-fraction mean, tracked over time without classifying
    content. rho=(mu-L)^2/[v+(mu-L)^2] is a moment-based effective occupancy,
    not a speech probability. T=L+sqrt(v+(mu-L)^2) avoids division at rho=0.
    A sliding sorted window makes the lower-tail calculation O(radius) per bin.
    """
    rows,bins = clipped.shape
    radius=max(1,int(SURFACE_FREQUENCY_HZ*2.*(GUIDE_FFT-1)/rate))
    reference[:] = floor
    for t in range(rows):
        left=0;right=min(bins,radius+1);count=right
        for j in range(count):scratch[j]=clipped[t,j]
        sort_values(scratch[:count], (scratch[:count]).size, sort_stack)
        for b in range(bins):
            mass=SURFACE_LOW_FRACTION*count;whole=int(mass);fraction=mass-whole
            total=0.
            for j in range(whole):total+=scratch[j]
            if fraction>0.:total+=fraction*scratch[whole]
            low[t,b]=total/mass
            nl=max(0,b+1-radius);nr=min(bins,b+radius+2)
            if nl>left:
                value=clipped[t,left];j=0
                while j<count-1 and scratch[j]!=value:j+=1
                for k in range(j,count-1):scratch[k]=scratch[k+1]
                count-=1
            if nr>right:
                value=clipped[t,right];j=count
                while j>0 and scratch[j-1]>value:
                    scratch[j]=scratch[j-1];j-=1
                scratch[j]=value;count+=1
            left,right=nl,nr
    alpha=1.-np.exp(-MASK_HOP/(rate*SURFACE_TRACK_SECONDS))
    continuous=meta[0]!=0. and origin==meta[1]+BLOCK_SIZE
    for b in range(bins):
        lower=checkpoint[b] if continuous else low[0,b]
        threshold=checkpoint[bins+b] if continuous else 0.
        for t in range(rows):
            scale=max(means[t,b],np.sqrt(variances[t,b]))
            lower=max(0.,lower+alpha*min(scale,max(-scale,low[t,b]-lower)))
            lower=min(lower,means[t,b])
            difference=max(0.,means[t,b]-lower)
            second=variances[t,b]+difference*difference
            low[t,b]=lower
            occupancy[t,b]=difference*difference/second if second>0. else 1.
            target=lower+np.sqrt(second)
            threshold=target if t==0 and not continuous else max(0.,threshold+alpha*min(scale,max(-scale,target-threshold)))
            floor[t,b]=threshold
            if t==BLOCK_SIZE//MASK_HOP-1:
                checkpoint[b],checkpoint[bins+b]=lower,threshold
    meta[0],meta[1]=1.,origin


@njit(F64(MAT, MAT, MAT, VEC, F64, VEC), nogil=True, cache=False)
def local_distribution_shape(data, mean, score, scratch, rate, sort_stack):
    """Cleanup's sorted-logit comparison on locally equalized observations.

    Use three ACTUAL guide centers, not interpolated time samples. Division by
    the bounded local mean removes receiver-envelope scale before pooling;
    no passband, quiet-frame label or temporal minimum enters this comparison.
    Scores are descriptive, not calibrated noise probabilities.
    """
    rows,bins=data.shape
    radius=min(bins-1,max(1,int(SHAPE_RADIUS_HZ*2.*(GUIDE_FFT-1)/rate)))
    width=2*radius+1;n=3*width;step=max(1,GUIDE_HOP//MASK_HOP)
    values=scratch[:n];logit=scratch[n:2*n]
    logistic(logit,n)
    values[:]=0.;values[n-1]=1.
    maximum=1.-correlation(values,logit,n)
    for t in range(0,rows,step):
        b=0
        while b<bins:
            k=0
            for dt in range(-1,2):
                row=min(((rows-1)//step)*step,max(0,t+dt*step))
                for db in range(-radius,radius+1):
                    col=b+db
                    if col<0:col=-col
                    if col>=bins:col=2*bins-2-col
                    values[k]=data[row,col]/mean[row,col] if mean[row,col]>1e-24 else 0.
                    k+=1
            sort_values(values, (values).size, sort_stack)
            score[t,b]=max(0.,min(maximum,1.-correlation(values,logit,n)))
            if b==bins-1:break
            b=min(bins-1,b+SHAPE_STRIDE)
        for col in range(bins):
            left=(col//SHAPE_STRIDE)*SHAPE_STRIDE;right=min(bins-1,left+SHAPE_STRIDE)
            f=(col-left)/max(1,right-left)
            score[t,col]=(1.-f)*score[t,left]+f*score[t,right]
    last=((rows-1)//step)*step
    for t in range(rows):
        lo=min(last,(t//step)*step);hi=min(last,lo+step);f=(t-lo)/step
        if t!=lo:
            for b in range(bins):score[t,b]=(1.-f)*score[lo,b]+f*score[hi,b]
    return maximum


@njit(PAIR(MAT, MAT, MAT, MAT, MAT, MAT, MAT, VEC, F64, F64, BOOL, VEC), nogil=True, cache=False)
def shape_surface_peaks(data, observations, mask, low, reference, contour, score, scratch, rate, maximum, publish, sort_stack):
    """Local counterpart of fast_peaks' gate and row/context blend.

    The contextual pair is low reference + residual RMS; current-row pairs
    use the same bottom-fraction reference. Preserve baseline's exponential
    blend and second-pass in-place behavior. Counts/squelch are untouched.
    """
    rows,bins=data.shape
    radius=max(1,int(SURFACE_FREQUENCY_HZ*2.*(GUIDE_FFT-1)/rate))
    total=accepted=0.
    for t in range(rows):
        # Copy before writing: the second pass aliases data and mask.
        scratch[:bins]=observations[t]
        left=0;right=min(bins,radius+1);n=right
        for k in range(n):scratch[bins+k]=scratch[k]
        sort_values(scratch[bins:bins+n], (scratch[bins:bins+n]).size, sort_stack)
        for b in range(bins):
            raw=score[t,b];smooth=(score[max(0,t-1),b]+raw+score[min(rows-1,t+1),b])/3.
            mass=SURFACE_LOW_FRACTION*n;whole=int(mass);fraction=mass-whole
            rm=0.
            for k in range(whole):rm+=scratch[bins+k]
            if fraction>0.:rm+=fraction*scratch[bins+whole]
            rm/=mass
            ss=0.
            for k in range(left,right):ss+=(scratch[k]-rm)**2
            ra=np.sqrt(ss/n)
            factor=max(0.,1.-raw/max(maximum,1e-24))
            gm=low[t,b]*factor;ga=max(0.,reference[t,b]-low[t,b])*factor
            wm=np.exp(-.5*abs(gm-rm));wa=np.exp(-.5*abs(ga-ra))
            threshold=rm*wm+gm*(1.-wm)+MASK_STRENGTH*(ra*wa+ga*(1.-wa))
            if publish:contour[t,b]=threshold
            total+=threshold
            # Move the sorted neighborhood once; avoid a sort at every bin.
            nl=max(0,b+1-radius);nr=min(bins,b+radius+2)
            if nl>left:
                value=scratch[left];j=0
                while j<n-1 and scratch[bins+j]!=value:j+=1
                for k in range(j,n-1):scratch[bins+k]=scratch[bins+k+1]
                n-=1
            if nr>right:
                value=scratch[right];j=n
                while j>0 and scratch[bins+j-1]>value:
                    scratch[bins+j]=scratch[bins+j-1];j-=1
                scratch[bins+j]=value;n+=1
            left,right=nl,nr
            # Same fallback as unsquelched Cleanup, applied to local shape.
            admission=1. if smooth>VAD_THRESHOLD or raw>=VAD_THRESHOLD else 0.
            if SHAPE_SOFT_ADMISSION:
                x=min(1.,max(0.,max(smooth,raw)/VAD_THRESHOLD))
                admission=x*x*(3.-2.*x)
            if admission<=0.:continue
            if data[t,b]>threshold:
                mask[t,b]=admission;accepted+=admission
    return total/(rows*bins),accepted/(rows*bins)


@njit(PAIR(MAT, MAT, MAT, I64), nogil=True, cache=False)
def surface_peaks(data, mask, floor, nb):
    """Use the same measured surface in both recursive Cleanup passes."""
    selected = total = 0.
    for t in range(data.shape[0]):
        for b in range(nb):
            threshold = floor[t,b]
            value = data[t,b]
            mask[t,b] = 1. if value > threshold else 0.
            selected += mask[t,b]
            total += threshold
    count = data.shape[0]*nb
    return total/count,selected/count


@njit(PAIR(MAT, MAT, MAT, I64), nogil=True, cache=False)
def surface_residual_peaks(data, mask, floor, nb):
    """Audit: change ONLY occupancy's failed-cell second-pass assignment."""
    selected=total=0.
    for t in range(data.shape[0]):
        for b in range(nb):
            threshold=floor[t,b]
            if data[t,b]>threshold:mask[t,b]=1.;selected+=1.
            total+=threshold
    count=data.shape[0]*nb
    return total/count,selected/count


@njit(VOID(MAT, I64, VEC, I64), nogil=True, cache=False)
def audit_gain_stage(mask, nb, state, offset):
    """Emitted gain morphology only; no policy or audio modification."""
    total=zeros=soft=ones=tv=fv=power=maximum=0.
    for t in range(FIRST,FIRST+EMIT):
        for b in range(nb):
            g=min(1.,max(0.,mask[t,b]));total+=g;power+=g*g;maximum=max(maximum,g)
            if g==0.:zeros+=1.
            elif g==1.:ones+=1.
            else:soft+=1.
            if t>FIRST:tv+=abs(g-min(1.,max(0.,mask[t-1,b])))
            if b>0:fv+=abs(g-min(1.,max(0.,mask[t,b-1])))
    count=float(EMIT*nb)
    state[offset]=total/count;state[offset+1]=zeros/count;state[offset+2]=soft/count;state[offset+3]=ones/count
    state[offset+4]=tv/max(1.,(EMIT-1)*nb);state[offset+5]=fv/max(1.,EMIT*(nb-1))
    state[offset+6]=power/count;state[offset+7]=maximum


@njit(I32(CUBE, CUBE, MAT, VEC), nogil=True, cache=False)
def cleanup_registered(analysis, content, all_state, config):
    state = all_state[0]
    sort_stack = state[SORT_STACK_OFFSET:SORT_STACK_OFFSET + SORT_STACK_SIZE]
    half_bin = .5 if BINS == FFT_SIZE//2 else 0.
    if CLEANUP_POPULATION != "receiver" or REGISTERED_FLOOR in ("zero_crossing", "variance_surface", "occupancy_surface"):
        synth_nb, nb = BINS, GUIDE_BINS
    else:
        synth_nb = min(BINS, max(4, int(np.floor(config[1]/(config[0]/FFT_SIZE) + 1.5 - half_bin))))
        if config[1] > (GUIDE_BINS-1)*config[0]/(2*(GUIDE_FFT-1)):
            return ERROR  # Enlarge GUIDE_BINS and Reload; never silently crop the band.
        nb = min(GUIDE_BINS, max(4, int(np.floor(config[1]/(config[0]/(2*(GUIDE_FFT-1))) + 1.5))))
    squelch = config[2] != 0.
    # All working arrays are views of this entity's persistent fixed storage.
    offset = 64
    mag = state[offset:offset + MASK_PLANE].reshape((MASK_FRAMES, GUIDE_BINS)); offset += MASK_PLANE
    smoothed = state[offset:offset + MASK_PLANE].reshape((MASK_FRAMES, GUIDE_BINS)); offset += MASK_PLANE
    mask = state[offset:offset + MASK_PLANE].reshape((MASK_FRAMES, GUIDE_BINS)); offset += MASK_PLANE
    vertical = state[offset:offset + GUIDE_PAD_PLANE].reshape((GUIDE_PAD_T, GUIDE_PAD_B)); offset += GUIDE_PAD_PLANE
    horizontal = state[offset:offset + GUIDE_PAD_PLANE].reshape((GUIDE_PAD_T, GUIDE_PAD_B)); offset += GUIDE_PAD_PLANE
    values = state[offset:offset + MASK_PLANE]; offset += MASK_PLANE
    deviations = state[offset:offset + MASK_PLANE]; offset += MASK_PLANE
    raw = state[offset:offset + MASK_FRAMES]; offset += MASK_FRAMES
    smooth = state[offset:offset + MASK_FRAMES]; offset += MASK_FRAMES
    detected = state[offset:offset + MASK_FRAMES]; offset += MASK_FRAMES
    log1 = state[offset:offset + GUIDE_BINS]; offset += GUIDE_BINS
    log3 = state[offset:offset + 3 * GUIDE_BINS]; offset += 3 * GUIDE_BINS
    scratch = state[offset:]
    population_bins=nb
    guide = all_state[GUIDE_SLOT,GUIDE_OFFSET:GUIDE_OFFSET+GUIDE_PLANE].reshape((GUIDE_FRAMES,GUIDE_BINS))
    if CLEANUP_POPULATION == "rolloff":
        envelope=scratch[:GUIDE_BINS]
        prefix=scratch[GUIDE_BINS:GUIDE_BINS+485].reshape((97,5))
        work=scratch[GUIDE_BINS+485:2*GUIDE_BINS+677]
        diagnostic=scratch[2*GUIDE_BINS+677:2*GUIDE_BINS+685]
        population_bins=population_support(guide,config[0],2.*(GUIDE_FFT-1),envelope,prefix,work,diagnostic, sort_stack)
        state[28]=population_bins;state[29]=diagnostic[0];state[30]=diagnostic[1]
    elif CLEANUP_POPULATION == "oracle":
        population_bins=min(GUIDE_BINS,max(4,int(np.floor(config[1]/(config[0]/(2*(GUIDE_FFT-1)))+1.5))))
        state[28]=population_bins;state[29]=config[1];state[30]=0.
    # Reserved row 3 owns reference evidence scratch; row 2 remains immutable.
    evidence_storage = all_state[3]
    rmag = evidence_storage[64:64+REFERENCE_PLANE].reshape((REFERENCE_FRAMES,REFERENCE_BINS))
    ro = 64 + REFERENCE_PLANE
    rraw = evidence_storage[ro:ro+REFERENCE_FRAMES]; ro += REFERENCE_FRAMES
    rsmooth = evidence_storage[ro:ro+REFERENCE_FRAMES]; ro += REFERENCE_FRAMES
    rdetected = evidence_storage[ro:ro+REFERENCE_FRAMES]; ro += REFERENCE_FRAMES
    rlog1 = evidence_storage[ro:ro+REFERENCE_BINS]; ro += REFERENCE_BINS
    rlog3 = evidence_storage[ro:ro+3*REFERENCE_BINS]; ro += 3*REFERENCE_BINS
    reference_nb = REFERENCE_BINS if REGISTERED_FLOOR in ("zero_crossing", "variance_surface", "occupancy_surface") else min(REFERENCE_BINS,max(4,int(np.floor(config[1]/(config[0]/512)+1.5))))
    if CLEANUP_POPULATION != "receiver":
        edge=(population_bins-1)*config[0]/(2.*(GUIDE_FFT-1))
        reference_nb=min(REFERENCE_BINS,max(4,int(np.floor(edge/(config[0]/512.)+1.5))))
    # Parent publishes actual fixed 512/128 RFFT evidence, independent of synthesis.
    # Excluded bins must be zero for the original statistics conventions.
    for t in range(REFERENCE_FRAMES):
        for b in range(reference_nb,REFERENCE_BINS):
            rmag[t,b] = 0.
    active, rmaximum, rmaximum3 = reference_entropy(rmag,reference_nb,rraw,rsmooth,rdetected,rlog1,rlog3,evidence_storage[ro:], sort_stack)
    record_evidence(rsmooth,state,8,128)
    mag[:] = 0.
    smoothed[:] = 0.
    mask[:] = 0.
    guide = all_state[GUIDE_SLOT,GUIDE_OFFSET:GUIDE_OFFSET+GUIDE_PLANE].reshape((GUIDE_FRAMES,GUIDE_BINS))
    for t in range(MASK_FRAMES):
        x = t * MASK_HOP / GUIDE_HOP
        t0 = min(int(x), GUIDE_FRAMES - 1)
        t1 = min(t0 + 1, GUIDE_FRAMES - 1)
        fraction = min(1., x - t0)
        for b in range(nb):
            mag[t,b] = ((1.-fraction)*guide[t0,b] + fraction*guide[t1,b]) * GUIDE_MAGNITUDE_SCALE
    guide_active, maximum, maximum3 = guide_entropy(mag, population_bins, raw, smooth, detected, log1, log3, scratch, active, sort_stack)
    record_evidence(smooth,state,12,MASK_HOP)
    state[16:24] = 0.  # Per-block floor/threshold diagnostics.
    state[24:28] = 0.
    crossing_floor = 0.
    if REGISTERED_FLOOR == "zero_crossing":
        audio = all_state[3,AUDIO_HISTORY_OFFSET:AUDIO_HISTORY_OFFSET+CONTEXT_BLOCKS*BLOCK_SIZE]
        adjacent_mean, adjacent_count = crossing_neighbors(audio)
        crossing_floor = adjacent_mean * ZERO_CROSSING_GUIDE_SCALE
        state[24], state[25] = adjacent_mean, adjacent_count
        state[26], state[27] = crossing_floor, 1.
    state[0] += 1.  # Persistent segment counter, useful to other entities/tests.
    state[1] = 1. if active else 0.
    if squelch and not active and REGISTERED_FLOOR not in ("variance_surface", "occupancy_surface"):
        return SKIP
    initial = np.max(mag)
    if initial <= 0. and REGISTERED_FLOOR not in ("variance_surface", "occupancy_surface"):
        return SKIP if squelch else PASSTHROUGH
    sawtooth(mag, smoothed, nb, scratch)
    man = atd = 0.
    surface = state[SURFACE_OFFSET:SURFACE_OFFSET+MASK_PLANE].reshape((MASK_FRAMES,GUIDE_BINS)) if REGISTERED_FLOOR in ("variance_surface", "occupancy_surface") else mask
    if REGISTERED_FLOOR in ("variance_surface", "occupancy_surface"):
        mu = surface
        var = state[SURFACE_OFFSET+MASK_PLANE:SURFACE_OFFSET+2*MASK_PLANE].reshape((MASK_FRAMES,GUIDE_BINS))
        surface = state[SURFACE_OFFSET+2*MASK_PLANE:SURFACE_OFFSET+3*MASK_PLANE].reshape((MASK_FRAMES,GUIDE_BINS))
        clipped = state[SURFACE_OFFSET+3*MASK_PLANE:SURFACE_OFFSET+4*MASK_PLANE].reshape((MASK_FRAMES,GUIDE_BINS))
        checkpoint = state[SURFACE_CHECKPOINT:SURFACE_CHECKPOINT+3*GUIDE_BINS].reshape((3,GUIDE_BINS))
        meta = state[SURFACE_CHECKPOINT+3*GUIDE_BINS:SURFACE_CHECKPOINT+3*GUIDE_BINS+2]
        variance_surface(smoothed,mu,var,surface,clipped,checkpoint,meta,scratch,config[0],config[4], sort_stack)
        if REGISTERED_FLOOR == "occupancy_surface":
            low=state[OCCUPANCY_OFFSET:OCCUPANCY_OFFSET+MASK_PLANE].reshape((MASK_FRAMES,GUIDE_BINS))
            occupancy=state[OCCUPANCY_OFFSET+MASK_PLANE:OCCUPANCY_OFFSET+2*MASK_PLANE].reshape((MASK_FRAMES,GUIDE_BINS))
            reference=state[OCCUPANCY_OFFSET+2*MASK_PLANE:OCCUPANCY_OFFSET+3*MASK_PLANE].reshape((MASK_FRAMES,GUIDE_BINS))
            cp=state[OCCUPANCY_OFFSET+3*MASK_PLANE:OCCUPANCY_OFFSET+3*MASK_PLANE+2*GUIDE_BINS]
            cm=state[OCCUPANCY_OFFSET+3*MASK_PLANE+2*GUIDE_BINS:OCCUPANCY_OFFSET+3*MASK_PLANE+2*GUIDE_BINS+2]
            occupancy_surface(clipped,mu,var,surface,low,occupancy,reference,cp,cm,scratch,config[0],config[4], sort_stack)
        if OCCUPANCY_SHAPE and REGISTERED_FLOOR == "occupancy_surface":
            shape=state[SHAPE_OFFSET:SHAPE_OFFSET+MASK_PLANE].reshape((MASK_FRAMES,GUIDE_BINS))
            shape_maximum=local_distribution_shape(mag,mu,shape,scratch,config[0], sort_stack)
            reference[:]=surface  # Preserve the unweighted occupancy contour.
            state[18],state[19]=shape_surface_peaks(smoothed,clipped,mask,low,reference,surface,shape,scratch,config[0],shape_maximum,True, sort_stack)
        else:
            state[18],state[19] = surface_peaks(smoothed,mask,surface,nb)
    elif REGISTERED_FLOOR == "zero_crossing":
        if state[25] == 0.:
            return PASSTHROUGH  # No observation: do not invent a noise floor.
        state[18], state[19] = crossing_peaks(smoothed, mask, nb, crossing_floor)
    else:
        man, atd = statistics(smoothed, population_bins, values, deviations)
        state[16], state[17] = man, atd
        state[18], state[19] = population_peaks(smoothed, mask, nb, raw, detected, maximum, squelch, man, atd, values, deviations, population_bins)
    if CLEANUP_AUDIT_TRACE:audit_gain_stage(mask,nb,state,32)
    for t in range(MASK_FRAMES):
        for b in range(nb):
            if mask[t, b] == 0.:
                mag[t, b] = 0.
    multiplier = min(1., np.max(mag) / initial) if initial > 0. else 0.
    if REGISTERED_FLOOR == "legacy" and not (CLEANUP_AUDIT_BITS & 1):
        man, atd = statistics(mag, population_bins, values, deviations)
    sawtooth(mag, smoothed, nb, scratch)
    if REGISTERED_FLOOR in ("variance_surface", "occupancy_surface"):
        if OCCUPANCY_SHAPE and REGISTERED_FLOOR == "occupancy_surface":
            state[22],state[23]=shape_surface_peaks(smoothed,smoothed,smoothed,low,reference,surface,shape,scratch,config[0],shape_maximum,False, sort_stack)
        else:
            if CLEANUP_AUDIT_BITS & 128:
                state[22],state[23]=surface_residual_peaks(smoothed,smoothed,surface,nb)
            else:
                state[22],state[23] = surface_peaks(smoothed,smoothed,surface,nb)
    elif REGISTERED_FLOOR == "zero_crossing":
        state[22], state[23] = crossing_peaks(smoothed, smoothed, nb, crossing_floor)
    else:
        state[20], state[21] = man, atd
        if CLEANUP_AUDIT_BITS & 2:
            # Different output storage makes below-threshold cells zero while
            # retaining identical row/context threshold calculations and gates.
            # Dedicated plane keeps output separate from both sorting buffers.
            binary=state[SURFACE_OFFSET:SURFACE_OFFSET+MASK_PLANE].reshape((MASK_FRAMES,GUIDE_BINS))
            binary[:]=0.
            state[22],state[23]=population_peaks(smoothed,binary,nb,raw,detected,maximum,squelch,man,atd,values,deviations,population_bins)
            smoothed[:]=binary
        else:
            state[22], state[23] = population_peaks(smoothed, smoothed, nb, raw, detected, maximum, squelch, man, atd, values, deviations, population_bins)
    for t in range(MASK_FRAMES):
        for b in range(nb):
            mask[t, b] = min(mask[t, b], smoothed[t, b] * multiplier)
    if CLEANUP_AUDIT_TRACE:audit_gain_stage(mask,nb,state,40)
    sawtooth(mask, mask, nb, scratch)
    if recursive_smooth(mask, vertical, horizontal, nb, scratch) != 0:
        return ERROR
    if CLEANUP_AUDIT_TRACE:audit_gain_stage(mask,nb,state,48)
    if not squelch and REGISTERED_FLOOR == "legacy":
        for t in range(MASK_FRAMES):
            x = t * MASK_HOP / GUIDE_HOP
            t0 = min(int(x), GUIDE_FRAMES - 1)
            t1 = min(t0 + 1, GUIDE_FRAMES - 1)
            fraction = min(1., x - t0)
            for b in range(nb):
                mag[t,b] = ((1.-fraction)*guide[t0,b] + fraction*guide[t1,b]) * GUIDE_MAGNITUDE_SCALE
        man, atd = statistics(mag, population_bins, values, deviations)
        # Legacy extrema started at zero. Fix stale extrema across calls.
        maximum_value = np.max(mag)
        if maximum_value > 0.:
            man /= maximum_value
            atd /= maximum_value
    for t in range(MASK_FRAMES):
        for b in range(GUIDE_BINS):
            gain = mask[t,b] if b < nb else 0.
            # The guide is a full-context matrix; evidence and gain refer to
            # the same center. Subtracting the emitted-frame offset here used
            # an earlier interval's decision for the current interval.
            if not (CLEANUP_AUDIT_BITS & 16) and not squelch and b < nb and detected[t] == 0.:
                gain = max(gain, man*smooth[t]/maximum3 + atd*(1.-MASK_STRENGTH))
            if OCCUPANCY_SHAPE and REGISTERED_FLOOR == "occupancy_surface" and not squelch:
                e=(shape[max(0,t-1),b]+shape[t,b]+shape[min(MASK_FRAMES-1,t+1),b])/3.
                if e<=VAD_THRESHOLD and initial>0.:
                    gain=max(gain,low[t,b]/initial*min(1.,e/max(shape_maximum,1e-24)))
            mask[t,b] = min(1.,max(0.,gain))
    if CLEANUP_AUDIT_TRACE:audit_gain_stage(mask,nb,state,56)
    if squelch and not active:
        return SKIP
    apply_guide_gain(mask,content,synth_nb,half_bin)
    return PROCESS


class FilterBank:
    """Compiled chain; native owner supplies one independent storage row per slot.

    Static methods use @njit. No per-audio-call class allocation.
    The loader verifies every static method has a nopython specialization.
    """
    @staticmethod
    @njit(FILTER_SIGNATURE, nogil=True, cache=False)
    def run(analysis, content, state, config):
        result = PASSTHROUGH
        for slot in range(len(CHAIN)):
            if CHAIN[slot] == 0:
                if ANALYSIS_MODE == "registered":
                    action = cleanup_registered(analysis,content,state,config)
                else:
                    action = cleanup(analysis if slot == 0 else content, content, state[slot], config)
            else:
                return ERROR
            if action == SKIP or action == ERROR:
                return action
            if action == PROCESS:
                result = PROCESS
        return result


# Numba resolves the compiled dispatcher alias without a Python class lookup.
_run_bank = FilterBank.run


@njit(FILTER_SIGNATURE, nogil=True, cache=False)
def Filter(analysis, content, state, config):
    """Main native-callable segment entry point. Return the parent's disposition."""
    return _run_bank(analysis, content, state, config)
