// Standalone Windows host: deliberately has no Python or Cleanup imports.
// Run from any working directory, passing an absolute path to Cleanup.dll.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "cleanup_loader.h"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#define GET(name) auto name=reinterpret_cast<decltype(&::name)>(GetProcAddress(dll,#name)); if(!name){std::fprintf(stderr,"Missing %s\n",#name);return 2;}
int wmain(int argc,wchar_t** argv){
 if(argc<2)return 2;
 HMODULE dll=LoadLibraryExW(argv[1],nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
 if(!dll){std::fprintf(stderr,"LoadLibrary error %lu\n",GetLastError());return 2;}
 GET(cleanup_create) GET(cleanup_destroy) GET(cleanup_status) GET(cleanup_process)
 GET(cleanup_loader_error) GET(cleanup_reload) GET(cleanup_set_bypass) GET(cleanup_set_squelch)
 auto engine=cleanup_create(nullptr,nullptr,48000,2);
 if(!engine){char error[4096];cleanup_loader_error(error,sizeof(error));std::fprintf(stderr,"%s\n",error);return 3;}
 auto status=[&]{std::vector<char> b(cleanup_status(engine,nullptr,0));cleanup_status(engine,b.data(),uint32_t(b.size()));return std::string(b.data());};
 auto wait=[&]{for(int i=0;i<3600;++i){auto s=status();if(s.find("\"loading\":false")!=s.npos){std::puts(s.c_str());return s.find("\"success\":true")!=s.npos;}Sleep(100);}return false;};
 if(!wait()){cleanup_destroy(engine);return 4;}
 std::vector<float> x(8192*2),y(x.size());std::vector<double> timings;
 LARGE_INTEGER frequency,a,b;QueryPerformanceFrequency(&frequency);
 unsigned noise=7751;int blocks=argc>2?_wtoi(argv[2]):16;
 for(int block=0;block<blocks;++block){
  for(size_t j=0;j<x.size()/2;++j){noise=noise*1664525+1013904223;float v=float(.09*std::sin(2*3.141592653589793*713*(block*8192+j)/48000)+.02*(double(noise)/4294967296.-.5));x[2*j]=x[2*j+1]=v;}
  QueryPerformanceCounter(&a);int result=cleanup_process(engine,x.data(),y.data(),8192);QueryPerformanceCounter(&b);
  timings.push_back(double(b.QuadPart-a.QuadPart)/frequency.QuadPart);
  if(result||!std::all_of(y.begin(),y.end(),[](float v){return std::isfinite(v);})){cleanup_destroy(engine);return 5;}
 }
 std::puts(status().c_str());std::sort(timings.begin(),timings.end());
 std::printf("{\"median_seconds\":%.9f,\"peak_seconds\":%.9f,\"block_seconds\":%.9f}\n",timings[timings.size()/2],timings.back(),8192./48000);
 cleanup_set_bypass(engine,1);cleanup_set_squelch(engine,1);
 auto controls=status();if(controls.find("\"bypass\":true")==controls.npos||controls.find("\"squelch\":true")==controls.npos)return 6;
 if(cleanup_reload(engine)!=1||!wait()){cleanup_destroy(engine);return 7;}
 cleanup_destroy(engine);std::puts("bundle_host PASS");return 0;
}
