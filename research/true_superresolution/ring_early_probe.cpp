#include "ring_early.h"
#include "guide_peak.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
using Z=std::complex<double>;
constexpr int N=48,M=N*N;
int main(int argc,char** argv){
    if(argc!=2)return 2;
    std::ifstream f(argv[1],std::ios::binary|std::ios::ate);if(!f)return 2;
    size_t bytes=size_t(f.tellg());if(bytes%(M*sizeof(Z)))return 2;
    std::vector<Z> units(bytes/sizeof(Z));f.seekg(0);f.read(reinterpret_cast<char*>(units.data()),bytes);
    int charts=int(units.size()/M);if(charts<=9)return 2;
    std::vector<double> masks(7*M),fields(7*M);
    for(int y=0;y<N;++y)for(int x=0;x<N;++x){double fy=double(y<24?y:y-N)/N,fx=double(x<24?x:x-N)/N;
        for(int r=0;r<7;++r)masks[r*M+y*N+x]=std::exp(-.5*std::pow((std::hypot(fx,fy)-(.035+(.46-.035)*r/6))/.055,2));}
    cleanup_ring::RingForward48 full(masks.data());cleanup_ring_research::Pruned48 late(masks.data());cleanup_ring_research::Early48 early(masks.data());
    cleanup_ring_research::Peak out[7];int mismatch=0,ties=0,consensus=0;double pe=0,ce=0;size_t branches=0,rows=0;
    for(int t=0;t<charts;++t){const auto* u=units.data()+t*M;full.run(u,fields.data());
        for(bool allow:{false,true}){bool shortcut=early.run(u,out,allow);
            for(int r=0;r<7;++r){auto* field=fields.data()+r*M;int best;double comp;cleanup_phase_peak(field,N,best,comp);
                if(out[r].best!=best){if(std::abs(field[best]-field[out[r].best])<1e-12)++ties;else ++mismatch;}
                pe=std::max(pe,std::abs(out[r].peak-field[best]));
                if(!shortcut)ce=std::max(ce,std::abs(out[r].competitor-comp));
                if(shortcut&&best!=0)++mismatch;
                if(allow&&t>=9)rows+=out[r].rows;
            }
            if(allow&&t>=9){branches+=early.branches_completed;consensus+=shortcut;}
        }
    }
    std::vector<double> times[2];volatile double checksum=0;
    for(int repeat=0;repeat<14;++repeat)for(int step=0;step<2;++step){int method=(step+repeat)%2;auto start=std::chrono::steady_clock::now();
        for(int t=9;t<charts;++t){bool shortcut=method?early.run(units.data()+t*M,out,true):late.run(units.data()+t*M,out,true);
            for(const auto& p:out)checksum+=p.best+(shortcut?0.:p.competitor);}
        times[method].push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/(charts-9));}
    for(auto& v:times)std::sort(v.begin(),v.end());
    std::cout<<std::setprecision(17)<<"{\n  \"charts\": "<<charts<<",\n  \"peak_mismatches\": "<<mismatch<<",\n  \"near_tie_differences\": "<<ties
       <<",\n  \"max_peak_error\": "<<pe<<",\n  \"max_competitor_error\": "<<ce<<",\n  \"zero_consensus\": "<<consensus
       <<",\n  \"first_axis_branches_skipped\": "<<1.-double(branches)/(double(charts-9)*7*CLEANUP_EARLY_Q)
       <<",\n  \"final_outputs_skipped\": "<<1.-double(rows)/(double(charts-9)*7*48)
       <<",\n  \"late_us\": "<<times[0][7]<<",\n  \"early_us\": "<<times[1][7]<<",\n  \"speedup\": "<<times[0][7]/times[1][7]<<"\n}\n";
    return mismatch||ties||pe>1e-11||ce>1e-11?1:0;
}
