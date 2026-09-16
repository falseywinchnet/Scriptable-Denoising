At present, we have precisely three different kinds of the FFT- DIF, DIP, DIT.
All of them take a specific walk along the cube towards the frequency form.
They either gather in time or gather in frequency. But our new DIP does neither. 
It gathers in time AS it gathers in frequency. It is an agile form.
But it is presently our slowest FFT and sub-optimal. 

However, from these things we learn multiple facts. We learn different kinds of FFT are possible.  And we learn SORT orders can be customized. 
One: you can pay a cost to transfer order to chaos, and it will be economically sound.  This is technically a scatter, but it is not recorded as such. We will call this Distribution.
Two- you can pay a cost to transfer chaos to order, and it is often expensive. This is technically a gather, but the algorithm is called a scatter sort. We will call this Collection.
Three: all FFT must pay a cost to get information from X to X[N] that is partially paid by sort behaviors. You can pay it up front, as Distribution or Collection, or you can amortize it.
Stockham amortizes it, but requires pingpongs. our current DIF uses collection on the forward, and distribution on the inverse, and both can technically be skipped, if one can work in so-called “bruun residual order”.
Our current DIT uses Distribution on both forward and inverse. While it is lacking in some SSE/AVX optimization, it is powerful, at many sizes trumps the others solely because of this.
Our current DIP, in its quirky way, choses to say- no. I will do a partial collection at one end, and partial distribution at the other. I will reduce my disorder slightly at both ends,
And try to pay two cheaper sorts than one expensive one. 

This, of course, is a mistake. But it is a good and rewarding mistake, precisely because it tells us More Than One Way To Skin The FFT (is possible).

And it lends us operational and excellent nuances into what parts of the FFT are requisite and what can be concocted.
We can discern DIP from DIT from DIF phase walks and determine which is the most SIMD-agile, independent of the sorting needed.
And, we can with a choice of DIP/DIT/DIF walks choose a form that is best suited to simd on different platforms.
And, we can build multiple fft, that use different sort methods. We can build distribution-pure forms and they will be the fastest if you have to have the FFT.
And we can build ones that are collection-forward, distributed-inverse, with the matching behavior, and they will be the fastest if you just need to run pre-compiled filter, element wise.

We also suspect, but do not yet know for sure, if yet a third form is possible. The DIP almost achieves it, but does not.
The DIP packets are themselves interwoven and thus must be reconstructed. Our next work will be to characterize the three walks and different kinds of amortized sort.
We hypothesize another form of amortized sort based FFT. In this FFT, a partial Distribution is paid on the intake of the forward, and on the outlet of the inverse.
At the frequency-side, a partial Collection is needed on both sides- to arrive at FFT natural bin order and to return from it to the order inside our butterfly system.
Like the DIF residual form, we hope to be able to ignore it if one can work in internal bin order. But, we hope to be able, also, by partially paying the sort up front,
To have a segregated clockwise sort at the frequency end: such that if one only wants, for example, the first half or up to , say, the closest packet partition of the bins to N//2, only needs to do a partial sort.
Because the packets inside N//2 contain already all the bins inside N//2. Thus, we hope, operationally, for a form that, instead of paying 1.0 the cost of a Distribution form, can, 
If only some of the bins must be worked on, pay 0.75 or 0.80 of the same cost, thus achieving a practical engineered form.

at this moment, leaving the problem of the sort itself out, for the DIF/DIP/DIT walk itself(where we will assume the most optimal Distribution sort is used in all cases) what does their walk look like- and is it fundementally shaped the same from a simd perspective?

Right now DIF, DIT, and DIP are not merely three walks; they are three walk plus sort equilibria that distorts their very design-The walk is co-shaped by the sort strategy.. If you want to know whether the walks themselves have different SIMD economics, you need to redesign them under a common sort regime and watch what survives.
The common regime should be the DIT-style Distribution regime: external order is pushed into the internal coordinate system before the serious arithmetic begins. That means the transform starts by placing data into the walk’s preferred chaotic basis, not by letting the walk create chaos and then collecting it later. Then the core is judged as a core, not as a hidden sort machine.
So the useful objects are not current DIF, current DIT, current DIP. They are DIF-D, DIT-D, DIP-D: same algebraic transform, same endpoint policy, different decimation coordinate.
DIT-D already exists. It pays Distribution, then runs its merge walk.
DIF-D would be a non-native rewrite of DIF where the input is distributed into the residual coordinate system that the DIF would otherwise naturally produce at the end. Then the DIF walk is asked to operate inside that coordinate system without relying on final Collection. This may feel backwards because DIF’s current beauty is that it earns its residual order for free. But for the experiment, you deliberately give it DIT’s economic model and see whether the split walk still has an advantage.
DIP-D would be the harder one. Current DIP is not Distribution-pure. It uses phase packets and pays partial Collection and partial Distribution. To make DIP-D, the ingest sort has to place samples directly into the phase packet coordinate system that the diagonal walk wants, so the packet geometry is born before the arithmetic rather than negotiated at both ends. Then the forward should finish in a consistent internal frequency coordinate, not a half-cleared packet stream unless that same coordinate is declared the output basis.
That is probably days of brutal work, because the sort is not just a table. It changes which values are neighbors at each split, which pairs can share twiddles, and whether a span is self-contained. A fake common sort will not answer the question. It has to be a structure-preserving Distribution for each walk.
The comparison I ultimately want is something like this: fix the endpoint economy to Distribution, fix the output basis to either standard order or a declared internal order, fix SIMD width and terminal kernels, then compare only the induced walk. The measured question becomes: after the same kind of order-to-chaos payment, which of stage-sweep splitting, mirrored merging, and phase-packet diagonal motion becomes the most efficient simd strategy? 

Forward and inverse are not symmetric workloads. Forward RFFT is an N real sample to N/2 plus 1 complex-bin compression problem. Inverse RFFT is an N/2 plus 1 complex-bin expansion back to N real samples. Even when the algebra is inverse, the machine problem is not just reversed arithmetic. The dense side and sparse side swap, the boundary currency reverses, and the best walk can change.
So the isolated-walk ranking should be directional.
For forward RFFT, DIT-style mirrored merging is likely best if standard frequency output is required. It pays Distribution on the N-side, then merges into the N/2 frequency form with regular SIMD. It also matches the compression direction: many real samples collapse into fewer structured complex bins.
For inverse RFFT, DIF-style splitting may be naturally better. It starts from the N/2 frequency form and expands outward to N real samples. A split walk matches that direction: residues unfold into time structure. If the inverse can use Distribution-like output stores rather than Collection-like reads, the economics become even better.
DIP is the bidirectional hybrid. Its special promise is that it may be good when the workload is neither “give me all natural bins” nor “start from all natural bins,” but rather “touch selected frequency structure and return.” It can partially clear both sides, or keep frequency work in packet form. That makes it less likely to be the raw full-spectrum winner, but more likely to create a useful middle path for partial-band or internal-order workflows.
In conclusion from all of this, we look at what we have. What we observe is our DIF does not consistently win over the DIT when isolated to the inverse only, despite both being Distribution based IRFFT. So, the DIT is clearly the best, but not necessarily at Float size, only at Double.
The logistics of memory and compute pressure may support using the DIF for the float form, the DIT for the double. We haven’t finalized this yet. DIP is especially interesting here because its claimed virtue is local diagonal cells: contiguous streams, broadcast twiddles, same-address read/write, no interior permutes. That might be bad in double today because packet scheduling overhead dominates, but in float it could go either way. It could become excellent if its cell scales cleanly to wider lanes. Or it could become terrible if the packet machinery doubles address pressure and cannot feed the lanes. You do not know until native float DIP exists.

--- end commentary, library author --
