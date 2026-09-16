// Windows entry DLL: no import-time Python dependency and no loader-lock work.
// Keep the interpreter and engine module loaded for the lifetime of the host.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "cleanup_loader.h"
#include <string>
#include <cstring>
#include <algorithm>
namespace {
INIT_ONCE once=INIT_ONCE_STATIC_INIT;
HMODULE engine_module=nullptr;
std::string error,default_filter,default_runtime;
#define SYMBOLS(X) \
 X(cleanup_abi_version) X(cleanup_create) X(cleanup_destroy) \
 X(cleanup_process) X(cleanup_process_pair) X(cleanup_reload) \
 X(cleanup_set_bypass) X(cleanup_set_squelch) X(cleanup_set_harmonics) \
 X(cleanup_set_bandwidth) X(cleanup_status) X(cleanup_control_port) \
 X(cleanup_dip_transport) X(cleanup_recursive_smooth)
#define DECLARE(name) decltype(&name) impl_##name=nullptr;
SYMBOLS(DECLARE)
#undef DECLARE
std::string utf8(const std::wstring& s){
 int n=WideCharToMultiByte(CP_UTF8,0,s.data(),int(s.size()),nullptr,0,nullptr,nullptr);
 std::string out(n,'\0');WideCharToMultiByte(CP_UTF8,0,s.data(),int(s.size()),out.data(),n,nullptr,nullptr);return out;
}
BOOL CALLBACK initialize(PINIT_ONCE,void*,void**){
 try{
  HMODULE self=nullptr;
  if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_PIN,
       reinterpret_cast<LPCWSTR>(&initialize),&self))throw std::string("Cannot locate Cleanup.dll");
  wchar_t path[32768];DWORD n=GetModuleFileNameW(self,path,32768);
  if(!n||n==32768)throw std::string("Cannot resolve Cleanup.dll directory");
  std::wstring directory(path,n);directory.resize(directory.find_last_of(L"\\/"));
  auto python=directory+L"\\_python";
  const auto interpreter=python+L"\\python312.dll";
  if(HMODULE existing=GetModuleHandleW(L"python312.dll")){
   DWORD size=GetModuleFileNameW(existing,path,32768);
   if(!size||size==32768||CompareStringOrdinal(path,int(size),interpreter.c_str(),int(interpreter.size()),TRUE)!=CSTR_EQUAL)
    throw std::string("Another Python 3.12 runtime is already loaded from a different directory; use a separate Cleanup process");
  }
  // CPython extension dependencies also need these directories during imports.
  if(!AddDllDirectory(directory.c_str())||!AddDllDirectory(python.c_str()))
   throw std::string("Cannot register bundled runtime directories");
  const DWORD flags=LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
  if(!LoadLibraryExW(interpreter.c_str(),nullptr,flags))
   throw std::string("Cannot load bundled _python\\python312.dll (Windows error ")+std::to_string(GetLastError())+")";
  engine_module=LoadLibraryExW((directory+L"\\CleanupNative.dll").c_str(),nullptr,flags);
  if(!engine_module)throw std::string("Cannot load CleanupNative.dll (Windows error ")+std::to_string(GetLastError())+")";
#define RESOLVE(name) impl_##name=reinterpret_cast<decltype(impl_##name)>(GetProcAddress(engine_module,#name)); if(!impl_##name)throw std::string("Missing engine symbol: " #name);
  SYMBOLS(RESOLVE)
#undef RESOLVE
  if(impl_cleanup_abi_version()!=1)throw std::string("Incompatible Cleanup engine ABI");
  default_filter=utf8(directory+L"\\Filter.py");default_runtime=utf8(directory+L"\\runtime");
 }catch(const std::string& e){error=e;}catch(...){error="Cleanup loader initialization failed";}
 return TRUE;
}
bool ready(){InitOnceExecuteOnce(&once,initialize,nullptr,nullptr);return error.empty();}
}
extern "C" {
uint32_t cleanup_abi_version(){return 1;}
uint32_t cleanup_loader_error(char* out,uint32_t capacity){
 ready();if(out&&capacity){size_t n=std::min<size_t>(error.size(),capacity-1);std::memcpy(out,error.data(),n);out[n]=0;}return uint32_t(error.size()+1);
}
cleanup_engine* cleanup_create(const char* f,const char* r,double sr,uint32_t channels){return ready()?impl_cleanup_create(f?f:default_filter.c_str(),r?r:default_runtime.c_str(),sr,channels):nullptr;}
void cleanup_destroy(cleanup_engine* e){if(e)impl_cleanup_destroy(e);}
int cleanup_process(cleanup_engine* e,const float* x,float* y,uint32_t n){return e?impl_cleanup_process(e,x,y,n):-1;}
int cleanup_process_pair(cleanup_engine* e,const float* a,const float* x,float* y,uint32_t n){return e?impl_cleanup_process_pair(e,a,x,y,n):-1;}
int cleanup_reload(cleanup_engine* e){return e?impl_cleanup_reload(e):-1;}
void cleanup_set_bypass(cleanup_engine* e,int v){if(e)impl_cleanup_set_bypass(e,v);}
void cleanup_set_squelch(cleanup_engine* e,int v){if(e)impl_cleanup_set_squelch(e,v);}
void cleanup_set_harmonics(cleanup_engine* e,int v){if(e)impl_cleanup_set_harmonics(e,v);}
void cleanup_set_bandwidth(cleanup_engine* e,double v){if(e)impl_cleanup_set_bandwidth(e,v);}
uint32_t cleanup_status(cleanup_engine* e,char* out,uint32_t n){return e?impl_cleanup_status(e,out,n):cleanup_loader_error(out,n);}
uint16_t cleanup_control_port(cleanup_engine* e){return e?impl_cleanup_control_port(e):0;}
int32_t cleanup_dip_transport(const double* s,const double* d,const double* w,double* o,double* a,const double* c,const double* t,int64_t n,int64_t v,int64_t h){return ready()?impl_cleanup_dip_transport(s,d,w,o,a,c,t,n,v,h):-1;}
int32_t cleanup_recursive_smooth(double* m,double* v,double* h,double* s,int64_t f,int64_t b,int64_t n,int64_t tp,int64_t fp,int64_t i){return ready()?impl_cleanup_recursive_smooth(m,v,h,s,f,b,n,tp,fp,i):-1;}
}
