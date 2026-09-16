#pragma once
// Research only. Exact peak/competitor queries with certified row pruning.
// The common Hermitian spectrum is transformed along one axis. Hermitian
// symmetry makes every remaining 1D row a real trigonometric polynomial:
// C[x] <= (A[0].real + abs(A[24].real) + 2*sum(abs(A[1:24]))) / 48.
// An unfinished row whose bound is below the incumbent needs no final FFT.
#include "ring_forward.h"
#include "guide_fft.h"
#include <numeric>
#include <limits>

namespace cleanup_ring_research {
using Z=std::complex<double>;
using cleanup_ring::Pack;
using cleanup_ring::ComplexPack;
struct Peak { int best=0, rows=0; double peak=0.,competitor=0.; };
class Pruned48 {
    static constexpr int N=48,M=N*N,groups=(7+Pack::width-1)/Pack::width;
    cleanup_ring::Forward48 forward;
    CleanupGuideFFT inverse{N};
    std::vector<Pack> weights;
    std::vector<Z> partial;
    std::vector<double> rows;
    std::array<ComplexPack,N> line,output;
    struct Search {
        double upper[N]; int order[N]; bool visited[N];
    } searches[7];
    void evaluate(int ring,int y,Peak& out){
        auto* a=partial.data()+ring*M+y*N;
        for(int x=N/2+1;x<N;++x)a[x]=std::conj(a[N-x]);
        inverse.inverse_real(a,rows.data()+ring*M+y*N);
        searches[ring].visited[y]=true;++out.rows;
    }
    bool transpose;
public:
    explicit Pruned48(const double* masks,bool transposed=true)
      :weights(groups*M),partial(7*M),rows(7*M),transpose(transposed){
        for(int g=0;g<groups;++g)for(int j=0;j<M;++j){
            double w[Pack::width]{};
            for(int l=0;l<Pack::width;++l)if(g*Pack::width+l<7)w[l]=masks[(g*Pack::width+l)*M+j];
            weights[g*M+j]=Pack::load(w);
        }
    }
    // true means all seven winning shifts are zero: the final weighted shift
    // and dispersion are then zero regardless of energies/confidence. When
    // enabled, unused competitors are deliberately NaN, never fake evidence.
    bool run(const Z* unit,Peak* result,bool allow_zero_consensus=false){
        bool zero=true;for(int j=0;j<M;++j)if(unit[j]!=Z{}){zero=false;break;}
        if(zero){std::fill_n(result,7,Peak{});return allow_zero_consensus;}
        // Inverse along y, carrying four ring coefficients through the same
        // complex forward butterflies. Retain only the Hermitian x half.
        for(int g=0;g<groups;++g)for(int x=0;x<=N/2;++x){
            for(int y=0;y<N;++y){int j=y*N+x,source=transpose?x*N+y:j;const auto u=unit[source];
                line[y]={weights[g*M+j]*Pack(u.real()),weights[g*M+j]*Pack(-u.imag())};}
            forward.complex(line.data(),output.data());
            for(int y=0;y<N;++y){double re[Pack::width],im[Pack::width];
                (output[y].re*Pack(1./N)).store(re);(output[y].im*Pack(-1./N)).store(im);
                for(int l=0;l<Pack::width;++l)if(g*Pack::width+l<7)partial[(g*Pack::width+l)*M+y*N+x]={re[l],im[l]};
            }
        }
        for(int ring=0;ring<7;++ring){
            auto& out=result[ring];out=Peak{};
            auto& state=searches[ring];auto& upper=state.upper;auto& order=state.order;
            std::fill_n(state.visited,N,false);
            auto* spectra=partial.data()+ring*M;auto* values=rows.data()+ring*M;
            for(int y=0;y<N;++y){const auto* a=spectra+y*N;
                double bound=a[0].real()+std::abs(a[N/2].real());
                for(int x=1;x<N/2;++x)bound+=2*std::sqrt(std::norm(a[x]));
                upper[y]=bound/N+1e-12*std::max(1.,std::abs(bound/N));order[y]=y;
            }
            std::sort(order,order+N,[&](int a,int b){return upper[a]>upper[b];});
            out.peak=-std::numeric_limits<double>::infinity();int best=M,py=0,px=0;
            for(int y:order){if(upper[y]<out.peak)break;
                evaluate(ring,y,out);
                for(int x=0;x<N;++x){double v=values[y*N+x];int index=transpose?x*N+y:y*N+x;
                    if(v>out.peak||(v==out.peak&&index<best)){out.peak=v;best=index;py=y;px=x;}}
            }
            out.best=best;
        }
        if(allow_zero_consensus&&std::all_of(result,result+7,[](const Peak& p){return p.best==0;})){
            for(int r=0;r<7;++r)result[r].competitor=std::numeric_limits<double>::quiet_NaN();
            return true;
        }
        for(int ring=0;ring<7;++ring){
            auto& out=result[ring];auto& state=searches[ring];auto& upper=state.upper;auto& order=state.order;
            auto& visited=state.visited;const auto* values=rows.data()+ring*M;
            int py=transpose?out.best%N:out.best/N,px=transpose?out.best/N:out.best%N;
            auto row_competitor=[&](int y){
                int dy=std::abs(y-py);bool all=std::min(dy,N-dy)>3;
                for(int x=0;x<N;++x){int dx=std::abs(x-px);
                    if(all||std::min(dx,N-dx)>3)out.competitor=std::max(out.competitor,values[y*N+x]);}
            };
            out.competitor=-std::numeric_limits<double>::infinity();
            for(int y=0;y<N;++y)if(visited[y])row_competitor(y);
            for(int y:order){if(upper[y]<out.competitor)break;
                if(!visited[y]){evaluate(ring,y,out);row_competitor(y);}}
        }
        return false;
    }
};
} // namespace cleanup_ring_research
