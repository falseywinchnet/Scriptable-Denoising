// Research-only statement clock; never linked into the distributed audio DLL.
#include <chrono>
#include <cstdint>
namespace {
uint64_t elapsed[256]{}, calls[256]{};
thread_local int previous=-1;
thread_local std::chrono::steady_clock::time_point started;
}
extern "C" void profile_mark(int slot) {
    const auto now=std::chrono::steady_clock::now();
    if(previous>=0){elapsed[previous]+=std::chrono::duration_cast<std::chrono::nanoseconds>(now-started).count();++calls[previous];}
    previous=slot;started=now;
}
extern "C" void profile_reset(){for(int i=0;i<256;++i)elapsed[i]=calls[i]=0;}
extern "C" uint64_t profile_elapsed(int slot){return elapsed[slot];}
extern "C" uint64_t profile_calls(int slot){return calls[slot];}
