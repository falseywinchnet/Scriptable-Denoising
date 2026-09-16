// Compare all exact peak/competitor decisions; time including bound/search cost.
#include "ring_pruned.h"
#include "guide_peak.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
using Z=std::complex<double>;
constexpr int N=48,M=N*N;
int main(int argc,char** argv){
    if(argc!=2){std::cerr<<"usage: ring_pruning_probe units.bin\n";return 2;}
    std::ifstream f(argv[1],std::ios::binary|std::ios::ate);
    if(!f){std::cerr<<"Cannot read spectra\n";return 2;}
    size_t bytes=size_t(f.tellg());if(bytes%(M*sizeof(Z))){std::cerr<<"Invalid spectra\n";return 2;}
    std::vector<Z> units(bytes/sizeof(Z));f.seekg(0);f.read(reinterpret_cast<char*>(units.data()),bytes);
    const int charts=int(units.size()/M);
    std::vector<double> masks(7*M),fields(7*M);
    for(int y=0;y<N;++y)for(int x=0;x<N;++x){
        double fy=double(y<N/2?y:y-N)/N,fx=double(x<N/2?x:x-N)/N,r=std::hypot(fx,fy);
        for(int j=0;j<7;++j)masks[j*M+y*N+x]=std::exp(-.5*std::pow((r-(.035+(.46-.035)*j/6))/.055,2));
    }
    cleanup_ring::RingForward48 full(masks.data());cleanup_ring_research::Pruned48 pruned(masks.data());
    cleanup_ring_research::Peak answers[7];
    int mismatch=0,numerical_ties=0,consensus=0;double peak_error=0,comp_error=0;size_t rows=0,consensus_rows=0;
    for(int t=0;t<charts;++t){auto* u=units.data()+t*M;full.run(u,fields.data());pruned.run(u,answers);
        for(int j=0;j<7;++j){int best;double competitor;auto* field=fields.data()+j*M;cleanup_phase_peak(field,N,best,competitor);
            if(best!=answers[j].best){if(std::abs(field[best]-field[answers[j].best])<1e-12)++numerical_ties;else ++mismatch;}
            peak_error=std::max(peak_error,std::abs(field[best]-answers[j].peak));comp_error=std::max(comp_error,std::abs(competitor-answers[j].competitor));
            if(t>=9)rows+=answers[j].rows;
        }
        bool skipped=pruned.run(u,answers,true);
        for(int j=0;j<7;++j){int best;double competitor;cleanup_phase_peak(fields.data()+j*M,N,best,competitor);
            if(skipped&&best!=0){std::cerr<<"Invalid zero-consensus shortcut\n";return 1;}
            if(t>=9)consensus_rows+=answers[j].rows;
        }
        if(t>=9&&skipped)++consensus;
    }
    std::vector<double> times[3];volatile double checksum=0.;
    for(int repeat=0;repeat<12;++repeat)for(int step=0;step<3;++step){int method=(step+repeat)%3;
        auto start=std::chrono::steady_clock::now();
        for(int t=9;t<charts;++t){auto* u=units.data()+t*M;
            if(method==0){full.run(u,fields.data());for(int j=0;j<7;++j){int best;double c;cleanup_phase_peak(fields.data()+j*M,N,best,c);checksum+=best+c;}}
            else{bool skipped=pruned.run(u,answers,method==2);for(const auto& a:answers)checksum+=a.best+(skipped?0.:a.competitor);}
        }
        times[method].push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/(charts-9));
    }
    for(auto& t:times)std::sort(t.begin(),t.end());
    std::cout<<std::setprecision(17)<<"{\n  \"charts\": "<<charts<<",\n  \"peak_mismatches\": "<<mismatch<<",\n  \"near_tie_location_differences\": "<<numerical_ties
        <<",\n  \"max_peak_error\": "<<peak_error<<",\n  \"max_competitor_error\": "<<comp_error
        <<",\n  \"speech_outputs_skipped_fraction\": "<<1.-double(rows)/(double(charts-9)*7*N)
        <<",\n  \"speech_zero_consensus_charts\": "<<consensus
        <<",\n  \"speech_outputs_skipped_with_consensus_fraction\": "<<1.-double(consensus_rows)/(double(charts-9)*7*N)
        <<",\n  \"fused_full_transform_and_search_us\": "<<times[0][6]<<",\n  \"pruned_transform_and_search_us\": "<<times[1][6]
        <<",\n  \"pruned_with_consensus_us\": "<<times[2][6]
        <<",\n  \"speedup\": "<<times[0][6]/times[1][6]
        <<",\n  \"speedup_with_consensus\": "<<times[0][6]/times[2][6]<<"\n}\n";
    return mismatch||numerical_ties||peak_error>1e-11||comp_error>1e-11?1:0;
}
