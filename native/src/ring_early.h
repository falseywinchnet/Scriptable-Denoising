#pragma once
// Experimental first-axis pruning. The 48 transform is paused after its
// three 16-point subtransforms; one bounded node represents y=t,t+16,t+32.
#include "ring_pruned.h"

namespace cleanup_ring_research {
#ifndef CLEANUP_EARLY_HEAP
#define CLEANUP_EARLY_HEAP 0
#endif
#ifndef CLEANUP_EARLY_ENERGY_BOUND
#define CLEANUP_EARLY_ENERGY_BOUND 0
#endif
inline Pack early_sqrt(Pack a){
#if defined(BRUUN_X86_128)
    for(auto& v:a.v)v=_mm_sqrt_pd(v);
#elif defined(BRUUN_NEON_128)
    for(auto& v:a.v)v=vsqrtq_f64(v);
#else
    for(auto& v:a.v)v=std::sqrt(v);
#endif
    return a;
}
struct EarlyNode {double bound;int code;};
inline bool operator<(const EarlyNode& a,const EarlyNode& b){return a.bound<b.bound;}
class Early48 {
#ifndef CLEANUP_EARLY_Q
#define CLEANUP_EARLY_Q 16
#endif
    static constexpr int N=48,M=N*N,Q=CLEANUP_EARLY_Q,L=N/Q,groups=(7+Pack::width-1)/Pack::width;
    static_assert(Q==12||Q==16,"Supported early radices are 12 and 16");
    CleanupGuideFFT inverse{N};
    std::vector<Pack> weights,energy_weights;
    std::vector<Z> stage,partial;
    std::vector<double> rows;
    std::array<ComplexPack,16> line,output;
    Z phase[L][Q];
    static void complex12(const ComplexPack* input,ComplexPack* out){
        ComplexPack residues[3][4];
        for(int r=0;r<3;++r)bruun::complex_dit4(input[r],input[r+3],input[r+6],input[r+9],residues[r]);
        constexpr double h=.866025403784438646763723170752936183;
        const Z twiddle[7]={{1,0},{h,-.5},{.5,-h},{0,-1},{-.5,-h},{-h,-.5},{-1,0}};
        const Z w(-.5,-h);
        for(int k=0;k<4;++k){const auto a=residues[0][k],b=residues[1][k]*twiddle[k],c=residues[2][k]*twiddle[2*k];
            out[k]=a+b+c;out[k+4]=a+b*w+c*std::conj(w);out[k+8]=a+b*std::conj(w)+c*w;}
    }
    struct Search {
        EarlyNode heap[48];int size=0;bool visited[N];
        double branch_bound[Q],row_bound[N];int order[Q];bool expanded[Q];
        void push(double bound,int code){
#if CLEANUP_EARLY_HEAP
            heap[size++]={bound,code};std::push_heap(heap,heap+size);
#else
            if(code<0){branch_bound[-1-code]=bound;order[-1-code]=-1-code;}else row_bound[code]=bound;
#endif
        }
        EarlyNode pop(){std::pop_heap(heap,heap+size);return heap[--size];}
    } searches[7];
    void expand(int ring,int t){
        ++branches_completed;
        searches[ring].expanded[t]=true;
        constexpr double h=.866025403784438646763723170752936183;
        const Z w(-.5,h);
        auto* source=stage.data()+ring*M;
        for(int x=0;x<=N/2;++x){
            if constexpr(L==4){
                const auto a=source[(t*L)*25+x]+source[(t*L+2)*25+x];
                const auto b=source[(t*L)*25+x]-source[(t*L+2)*25+x];
                const auto c=source[(t*L+1)*25+x]+source[(t*L+3)*25+x];
                const auto d0=source[(t*L+1)*25+x]-source[(t*L+3)*25+x];
                const Z d(-d0.imag(),d0.real());
                partial[ring*M+t*N+x]=a+c;partial[ring*M+(t+Q)*N+x]=b+d;
                partial[ring*M+(t+2*Q)*N+x]=a-c;partial[ring*M+(t+3*Q)*N+x]=b-d;
            }else{
                const auto a=source[(t*L)*25+x],b=source[(t*L+1)*25+x],c=source[(t*L+2)*25+x];
                partial[ring*M+t*N+x]=a+b+c;
                partial[ring*M+(t+Q)*N+x]=a+b*w+c*std::conj(w);
                partial[ring*M+(t+2*Q)*N+x]=a+b*std::conj(w)+c*w;
            }
        }
        for(int l=0;l<L;++l){int y=t+Q*l;auto* a=partial.data()+ring*M+y*N;
            double bound=a[0].real()+std::abs(a[N/2].real());
            for(int x=1;x<N/2;++x)bound+=2*std::sqrt(std::norm(a[x]));
            searches[ring].push(bound/N+1e-12,y);
        }
    }
    void evaluate(int ring,int y,Peak& result){
        auto* a=partial.data()+ring*M+y*N;
        for(int x=N/2+1;x<N;++x)a[x]=std::conj(a[N-x]);
        inverse.inverse_real(a,rows.data()+ring*M+y*N);
        searches[ring].visited[y]=true;++result.rows;
    }
public:
    int branches_completed=0;
    explicit Early48(const double* masks):weights(groups*M),energy_weights(groups*L*25),stage(7*M),partial(7*M),rows(7*M){
        for(int r=0;r<L;++r)for(int t=0;t<Q;++t)phase[r][t]=std::polar(1./N,-2*3.14159265358979323846*r*t/N);
        for(int g=0;g<groups;++g)for(int j=0;j<M;++j){double w[Pack::width]{};
            for(int l=0;l<Pack::width;++l)if(g*Pack::width+l<7)w[l]=masks[(g*Pack::width+l)*M+j];
            weights[g*M+j]=Pack::load(w);}
        // Weighted Cauchy-Schwarz: (sum |z|)^2 <= sum(W)*sum(|z|^2/W).
        // W is the positive radial mask's mass on each unfinished residue.
        // Fold multiplicity, total W and final normalization into a table.
        double envelope[7][L][25]{},total[7]{};
        for(int ring=0;ring<7;++ring)for(int r=0;r<L;++r)for(int x=0;x<=24;++x){
            for(int m=0;m<Q;++m)envelope[ring][r][x]+=masks[ring*M+(r+L*m)*N+x]/N;
            total[ring]+=envelope[ring][r][x]*(x==0||x==24?1.:2.);
        }
        for(int g=0;g<groups;++g)for(int r=0;r<L;++r)for(int x=0;x<=24;++x){double w[Pack::width]{};
            for(int l=0;l<Pack::width;++l){int ring=g*Pack::width+l;if(ring<7)w[l]=(x==0||x==24?1.:2.)*total[ring]/envelope[ring][r][x]/(N*N);}
            energy_weights[(g*L+r)*25+x]=Pack::load(w);
        }
    }
    bool run(const Z* unit,Peak* result,bool allow_zero_consensus=false){
        branches_completed=0;
        bool zero=true;for(int j=0;j<M;++j)if(unit[j]!=Z{}){zero=false;break;}
        if(zero){std::fill_n(result,7,Peak{});return allow_zero_consensus;}
        for(int ring=0;ring<7;++ring){searches[ring].size=0;std::fill_n(searches[ring].visited,N,false);std::fill_n(searches[ring].expanded,Q,false);result[ring]=Peak{};}
        for(int g=0;g<groups;++g){Pack bounds[Q];for(auto& b:bounds)b=Pack(0.);
            for(int x=0;x<=N/2;++x){Pack energy[Q];for(auto& e:energy)e=Pack(0.);
                for(int r=0;r<L;++r){
                    for(int m=0;m<Q;++m){int y=r+L*m,j=y*N+x;const auto u=unit[x*N+y];
                        line[m]={weights[g*M+j]*Pack(u.real()),weights[g*M+j]*Pack(-u.imag())};}
                    if constexpr(Q==16)bruun::complex_dit16_forward(line.data(),output.data());
                    else complex12(line.data(),output.data());
                    for(int t=0;t<Q;++t){auto a=output[t]*phase[r][t];double re[Pack::width],im[Pack::width];
                        a.re.store(re);(-a.im).store(im);
                        for(int l=0;l<Pack::width;++l)if(g*Pack::width+l<7)stage[(g*Pack::width+l)*M+(t*L+r)*25+x]={re[l],im[l]};
                        auto power=a.re*a.re+a.im*a.im;
#if CLEANUP_EARLY_ENERGY_BOUND == 2
                        bounds[t]=bounds[t]+power*energy_weights[(g*L+r)*25+x];
#elif CLEANUP_EARLY_ENERGY_BOUND == 1
                        energy[t]=energy[t]+power;
#else
                        bounds[t]=bounds[t]+early_sqrt(power)*Pack(x==0||x==24?1.:2.);
#endif
                    }
                }
#if CLEANUP_EARLY_ENERGY_BOUND == 1
                for(int t=0;t<Q;++t)bounds[t]=bounds[t]+early_sqrt(energy[t]*Pack(double(L)))*Pack(x==0||x==24?1.:2.);
#endif
            }
            for(int t=0;t<Q;++t){double bound[Pack::width];
#if CLEANUP_EARLY_ENERGY_BOUND == 2
                early_sqrt(bounds[t]).store(bound);
#else
                (bounds[t]*Pack(1./N)).store(bound);
#endif
                for(int l=0;l<Pack::width;++l)if(g*Pack::width+l<7)searches[g*Pack::width+l].push(bound[l]+1e-12,-1-t);}
        }
        for(int ring=0;ring<7;++ring){auto& state=searches[ring];auto& out=result[ring];out.peak=-std::numeric_limits<double>::infinity();int best=M;
#if CLEANUP_EARLY_HEAP
            while(state.size&&state.heap[0].bound>=out.peak){auto node=state.pop();
                if(node.code<0){expand(ring,-1-node.code);continue;}
                int y=node.code;evaluate(ring,y,out);const auto* v=rows.data()+ring*M+y*N;
                for(int x=0;x<N;++x){int index=x*N+y;if(v[x]>out.peak||(v[x]==out.peak&&index<best)){out.peak=v[x];best=index;}}
            }
#else
            std::sort(state.order,state.order+Q,[&](int a,int b){return state.branch_bound[a]>state.branch_bound[b];});
            for(int t:state.order){if(state.branch_bound[t]<out.peak)break;
                expand(ring,t);int ys[L];for(int l=0;l<L;++l)ys[l]=t+Q*l;
                std::sort(ys,ys+L,[&](int a,int b){return state.row_bound[a]>state.row_bound[b];});
                for(int y:ys){if(state.row_bound[y]<out.peak)break;
                    evaluate(ring,y,out);const auto* v=rows.data()+ring*M+y*N;
                    for(int x=0;x<N;++x){int index=x*N+y;if(v[x]>out.peak||(v[x]==out.peak&&index<best)){out.peak=v[x];best=index;}}}
            }
#endif
            out.best=best;
        }
        if(allow_zero_consensus&&std::all_of(result,result+7,[](const Peak& p){return p.best==0;})){
            for(int r=0;r<7;++r)result[r].competitor=std::numeric_limits<double>::quiet_NaN();return true;
        }
        for(int ring=0;ring<7;++ring){auto& state=searches[ring];auto& out=result[ring];int py=out.best%N,px=out.best/N;
            auto accept=[&](int y){int dy=std::abs(y-py);bool all=std::min(dy,N-dy)>3;
                for(int x=0;x<N;++x){int dx=std::abs(x-px);if(all||std::min(dx,N-dx)>3)out.competitor=std::max(out.competitor,rows[ring*M+y*N+x]);}};
            out.competitor=-std::numeric_limits<double>::infinity();
            for(int y=0;y<N;++y)if(state.visited[y])accept(y);
#if CLEANUP_EARLY_HEAP
            while(state.size&&state.heap[0].bound>=out.competitor){auto node=state.pop();
                if(node.code<0){expand(ring,-1-node.code);continue;}
                evaluate(ring,node.code,out);accept(node.code);
            }
#else
            for(int t:state.order){if(state.branch_bound[t]<out.competitor)break;
                if(!state.expanded[t])expand(ring,t);
                int ys[L];for(int l=0;l<L;++l)ys[l]=t+Q*l;
                std::sort(ys,ys+L,[&](int a,int b){return state.row_bound[a]>state.row_bound[b];});
                for(int y:ys){if(state.row_bound[y]<out.competitor)break;if(!state.visited[y]){evaluate(ring,y,out);accept(y);}}
            }
#endif
        }
        return false;
    }
};
} // namespace cleanup_ring_research
