#include "registered_guide.h"
#include "socket.hpp"
#include "cleanup.h"
#include <Python.h>
#include <bfft/stft.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using Callback = int32_t (*)(double*, double*, double*, double*);
std::mutex python_mutex;
struct Ref {
    PyObject* p;
    explicit Ref(PyObject* v=nullptr):p(v){}
    ~Ref(){Py_XDECREF(p);}
    Ref(const Ref&)=delete;
};
std::string python_error() {
    PyObject *type=nullptr, *value=nullptr, *trace=nullptr;
    PyErr_Fetch(&type,&value,&trace);
    PyErr_NormalizeException(&type,&value,&trace);
    Ref t(type), v(value), tb(trace), mod(PyImport_ImportModule("traceback"));
    if (mod.p) {
        Ref formatted(PyObject_CallMethod(mod.p,"format_exception","OOO", type?type:Py_None,value?value:Py_None,trace?trace:Py_None));
        if(formatted.p) {
            Ref sep(PyUnicode_FromString("")), joined(PyUnicode_Join(sep.p,formatted.p));
            if(joined.p) { const char* s=PyUnicode_AsUTF8(joined.p); if(s) return s; }
        }
    }
    PyErr_Clear();
    Ref str(value?PyObject_Str(value):nullptr);
    if(str.p) {const char* s=PyUnicode_AsUTF8(str.p); if(s) return s;}
    return "Python compilation failed (unable to format exception)";
}
void ensure_python() {
    if (!Py_IsInitialized()) {
        PyConfig config;
#ifdef _WIN32
        PyConfig_InitIsolatedConfig(&config);
        // Resolve the private runtime from this module, independently of the
        // host's working directory, PATH and any unrelated PYTHONHOME.
        std::wstring home;
        if(const wchar_t* override_home=_wgetenv(L"CLEANUP_PYTHONHOME"))home=override_home;
        else {
            HMODULE module=nullptr;wchar_t path[32768];
            if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                reinterpret_cast<LPCWSTR>(&ensure_python),&module)) {PyConfig_Clear(&config);throw std::runtime_error("Cannot locate Cleanup runtime");}
            DWORD n=GetModuleFileNameW(module,path,32768);
            if(!n||n==32768){PyConfig_Clear(&config);throw std::runtime_error("Cannot resolve Cleanup runtime path");}
            home.assign(path,n);home.resize(home.find_last_of(L"\\/"));home+=L"\\_python";
        }
        PyStatus configured=PyConfig_SetString(&config,&config.home,home.c_str());
        if(PyStatus_Exception(configured)){PyConfig_Clear(&config);throw std::runtime_error("Invalid private Python home");}
#else
        PyConfig_InitPythonConfig(&config);
        config.install_signal_handlers=0;
        // CLEANUP_PYTHONHOME selects an explicitly provisioned Python runtime.
        if (const char* home=std::getenv("CLEANUP_PYTHONHOME")) {
            PyStatus s=PyConfig_SetBytesString(&config,&config.home,home);
            if(PyStatus_Exception(s)) { PyConfig_Clear(&config); throw std::runtime_error("Invalid CLEANUP_PYTHONHOME"); }
        }
#endif
        PyStatus status=Py_InitializeFromConfig(&config);
        std::string error=PyStatus_Exception(status)?(status.err_msg?status.err_msg:"Python initialization failed"):"";
        PyConfig_Clear(&config);
        if(!error.empty()) throw std::runtime_error(error);
        PyEval_SaveThread();
    }
}
struct Gil { PyGILState_STATE state; Gil():state(PyGILState_Ensure()){} ~Gil(){PyGILState_Release(state);} };
std::string quoted(const std::string& s) {
    std::string out="\"";
    const char* hex="0123456789abcdef";
    for(unsigned char c:s) {
        if(c=='"'||c=='\\') {out+='\\';out+=c;}
        else if(c<32) {out+="\\u00";out+=hex[c>>4];out+=hex[c&15];}
        else out+=c;
    }
    return out+'"';
}
void check(bfft_status status) {if(status!=BFFT_OK) throw std::runtime_error(bfft_status_string(status));}
struct Plan {
    bfft_stft_plan* p=nullptr;
    ~Plan(){bfft_stft_plan_destroy(p);}
};
struct Geometry {
    size_t share_identical=0;
    size_t surface_shape=0,surface_shape_offset=0;
    size_t surface_extra=0,surface_low_offset=0,surface_occupancy_offset=0,surface_reference_offset=0;
    size_t surface=0,surface_offset=0,surface_mask_offset=0,surface_frames=0,surface_bins=0,surface_hop=0;
    size_t audio_history=0,audio_slot=0,audio_offset=0;
    size_t fft=0,hop=0,block=0,contexts=0,frames=0,bins=0,emit=0,first=0,slots=0,storage=0,transform=0;
    size_t guide=0,guide_fft=0,guide_bins=0,guide_frames=0,guide_hop=0,guide_slot=0,guide_offset=0;
    size_t guide_registration=1,mask_source_content=0;
    size_t viewer=0,viewer_autostart=0,view_custom=0,view_slot=0,view_offset=0;
    size_t view_bins=0,view_frames=0,view_first=0,view_emit=0,view_hop=0,view_frequency_denominator=0,view_half_bin=0;
};
struct Diagnostic {
    Geometry g;
    std::mutex mutex;
    std::vector<float> guide,filtered,input,surface_mean,surface_variance,surface_floor,surface_mask,surface_low,surface_occupancy,surface_reference,surface_shape;
    double evidence[8]{};
    double mask_stats[8]{};
    double mask_stages[32]{};
    double population_stats[4]{};
    double crossing_stats[4]{};
    double harmonic_edge=0.;
    double harmonic_stats[4]{};
    uint64_t generation=0,sequence=0;
    int action=0;
    bool bypass=false,squelch=false,gate_closed=false,valid=true;
    explicit Diagnostic(const Geometry& geometry):g(geometry),guide(g.view_emit*g.view_bins),filtered(g.emit*g.bins),input(g.emit*g.bins),
        surface_mean(g.surface?g.view_emit*g.view_bins:0),surface_variance(surface_mean.size()),
        surface_floor(surface_mean.size()),surface_mask(surface_mean.size()),
        surface_low(g.surface_extra?surface_mean.size():0),surface_occupancy(surface_low.size()),surface_reference(surface_low.size()),surface_shape(g.surface_shape?surface_mean.size():0){}
};
struct Channel {
    bool state_mirrors_first=true;
    Plan forward, inverse, reference_forward;
    void* guide_plan=nullptr;
    ~Channel(){cleanup_guide_destroy(guide_plan);}
    std::vector<double> analysis_history, content_history, incoming_analysis, incoming_content;
    std::vector<double> analysis, content, state, rendered, dry, guide_cache;
    std::vector<float> input_magnitude;
    std::vector<bfft_complex> packed, emitted, reference_packed;
    explicit Channel(const Geometry& g) {
        if(g.transform>1) throw std::runtime_error("Invalid synthesis transform");
        auto transform=static_cast<bfft_stft_transform>(g.transform);
        check(bfft_stft_plan_create(g.contexts*g.block,g.fft,g.hop,nullptr,transform,&forward.p));
        check(bfft_stft_plan_create(g.block,g.fft,g.hop,nullptr,transform,&inverse.p));
        if(bfft_stft_plan_bins(forward.p)!=g.bins || bfft_stft_plan_bins(inverse.p)!=g.bins)
            throw std::runtime_error("Filter bin count disagrees with bfft plans");
        analysis_history.resize(g.contexts*g.block); content_history.resize(g.contexts*g.block);
        incoming_analysis.resize(g.block); incoming_content.resize(g.block);
        analysis.resize(g.frames*g.bins*2); content.resize(g.frames*g.bins*2);
        state.resize(g.slots*g.storage); rendered.resize(g.block); dry.resize(g.block);
        packed.resize(g.frames*g.bins); emitted.resize(g.emit*g.bins);
        if(g.viewer)input_magnitude.resize(g.emit*g.bins);
        if(g.guide){
            check(bfft_stft_plan_create(24576,512,128,nullptr,BFFT_STFT_RFFT,&reference_forward.p));
            reference_packed.resize(192*257);guide_cache.resize(g.guide_bins*g.guide_frames);
            guide_plan=cleanup_guide_create_mode(g.guide_fft,g.guide_bins,g.guide_frames,g.guide_hop,0,int(g.guide_registration));
            if(!guide_plan)throw std::runtime_error("Registered guide plan construction failed");}
    }
};
struct Generation {
    Geometry g;
    Callback callback=nullptr;
    std::string token,hash;
    std::vector<std::unique_ptr<Channel>> channels;
    size_t input_pos=0,output_pos=0,output_count=0;
    uint64_t block_index=0;
    std::vector<float> output;
    std::shared_ptr<Diagnostic> diagnostic;
    // Destruction only occurs on the control worker or after audio is stopped.
    ~Generation() {
        if(token.empty() || !Py_IsInitialized()) return;
        std::lock_guard<std::mutex> lock(python_mutex); Gil gil;
        Ref module(PyImport_ImportModule("compile_filter"));
        if(module.p) {Ref result(PyObject_CallMethod(module.p,"release","s",token.c_str()));}
        PyErr_Clear();
    }
};
std::unique_ptr<Generation> compile_generation(const std::string& path,const std::string& runtime,uint32_t channels,double sample_rate,double bandwidth) {
    auto generation=std::make_unique<Generation>();
    {
        std::lock_guard<std::mutex> lock(python_mutex); ensure_python(); Gil gil;
        Ref directory(PyUnicode_DecodeFSDefault(runtime.c_str()));
        if(!directory.p || PyList_Insert(PySys_GetObject("path"),0,directory.p)<0) throw std::runtime_error(python_error());
        Ref module(PyImport_ImportModule("compile_filter"));
        if(!module.p) throw std::runtime_error(python_error());
        Ref result(PyObject_CallMethod(module.p,"compile_candidate","sKddK",path.c_str(),static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(&cleanup_dip_transport)),sample_rate,bandwidth,static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(&cleanup_recursive_smooth))));
        if(!result.p) throw std::runtime_error(python_error());
        auto integer=[&](const char* key) -> size_t {
            PyObject* value=PyDict_GetItemString(result.p,key);
            if(!value) throw std::runtime_error(std::string("Missing compiler metadata: ")+key);
            size_t n=PyLong_AsSize_t(value); if(PyErr_Occurred()) throw std::runtime_error(python_error()); return n;
        };
        auto str=[&](const char* key) -> std::string {
            PyObject* value=PyDict_GetItemString(result.p,key);
            const char* s=value?PyUnicode_AsUTF8(value):nullptr;
            if(!s) throw std::runtime_error("Invalid compiler string metadata"); return s;
        };
        generation->token=str("token"); generation->hash=str("sha256");
        auto& g=generation->g;
        g.share_identical=PyDict_GetItemString(result.p,"share_identical")?integer("share_identical"):0;
        g.mask_source_content=PyDict_GetItemString(result.p,"mask_source_content")?integer("mask_source_content"):0;
        g.guide_registration=PyDict_GetItemString(result.p,"guide_registration")?integer("guide_registration"):1;
        g.fft=integer("fft");g.hop=integer("hop");g.block=integer("block");g.contexts=integer("contexts");
        g.frames=integer("frames");g.bins=integer("bins");g.emit=integer("emit");g.first=integer("first");
        g.slots=integer("slots");g.storage=integer("storage");
        // Old compiler installations omit this metadata and remain RFFT.
        g.transform=PyDict_GetItemString(result.p,"transform")?integer("transform"):0;
        if(PyDict_GetItemString(result.p,"viewer")) {
            g.viewer=integer("viewer");g.viewer_autostart=integer("viewer_autostart");g.view_custom=integer("view_custom");
            g.view_slot=integer("view_slot");g.view_offset=integer("view_offset");g.view_bins=integer("view_bins");
            g.view_frames=integer("view_frames");g.view_first=integer("view_first");g.view_emit=integer("view_emit");
            g.view_hop=integer("view_hop");g.view_frequency_denominator=integer("view_frequency_denominator");g.view_half_bin=integer("view_half_bin");
        }
        if(PyDict_GetItemString(result.p,"guide")) {
            g.guide=integer("guide");g.guide_fft=integer("guide_fft");g.guide_bins=integer("guide_bins");
            g.guide_frames=integer("guide_frames");g.guide_hop=integer("guide_hop");
            g.guide_slot=integer("guide_slot");g.guide_offset=integer("guide_offset");
        }
        if(PyDict_GetItemString(result.p,"audio_history")) {
            g.audio_history=integer("audio_history");g.audio_slot=integer("audio_slot");g.audio_offset=integer("audio_offset");
        }
        if(PyDict_GetItemString(result.p,"surface") && integer("surface")) {
            g.surface=1;g.surface_offset=integer("surface_offset");g.surface_mask_offset=integer("surface_mask_offset");
            g.surface_frames=integer("surface_frames");g.surface_bins=integer("surface_bins");g.surface_hop=integer("surface_hop");
            if(PyDict_GetItemString(result.p,"surface_shape_offset")){g.surface_shape=1;g.surface_shape_offset=integer("surface_shape_offset");}
            if(PyDict_GetItemString(result.p,"surface_low_offset")){
                g.surface_extra=1;g.surface_low_offset=integer("surface_low_offset");g.surface_occupancy_offset=integer("surface_occupancy_offset");g.surface_reference_offset=integer("surface_reference_offset");
            }
        }
        generation->callback=reinterpret_cast<Callback>(integer("address"));
    }
    for(uint32_t i=0;i<channels;++i) generation->channels.push_back(std::make_unique<Channel>(generation->g));
    generation->output.resize(generation->g.block*channels);
    if(generation->g.viewer) generation->diagnostic=std::make_shared<Diagnostic>(generation->g);
    return generation;
}
}

struct cleanup_engine {
    std::string filter_path,runtime_dir;
    double sample_rate;
    uint32_t channel_count;
    std::atomic<bool> bypass{false},squelch{false},stop{false},loading{false},runtime_fault{false};
    std::atomic<double> bandwidth{3400.};
    std::atomic<double> fault_bandwidth{0.};
    std::atomic<bool> harmonics{false};
    std::atomic<uint64_t> processed{0},faults{0},last_ns{0},peak_ns{0};
    std::atomic<uint64_t> guide_ns{0},filter_ns{0};
    std::atomic<uint64_t> observation_ns{0},view_ns{0},registration_ns{0};
    std::atomic<uint64_t> stft_ns{0},publication_ns{0};
    std::atomic<uint32_t> filter_calls{0},guide_calls{0},shared_channels{0};
    std::mutex audio_mutex, status_mutex, worker_mutex;
    std::condition_variable wake;
    bool requested=true;
    uint64_t generation_id=0,attempt=0;
    bool success=false;
    std::string error="initializing";
    Geometry visible_geometry;
    size_t visible_guide_lanes=0;
    std::string visible_hash;
    std::string viewer_error;
    std::shared_ptr<Diagnostic> visible_diagnostic;
    std::unique_ptr<Generation> active;
    std::thread worker,server;
    socket_t listener=bad_socket;
    uint16_t port=0;
    cleanup_engine(std::string path,std::string runtime,double sr,uint32_t channels)
        :filter_path(std::move(path)),runtime_dir(std::move(runtime)),sample_rate(sr),channel_count(channels) {
        init_sockets();
        listener=socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);
        if(listener!=bad_socket) {
            sockaddr_in addr{};addr.sin_family=AF_INET;addr.sin_addr.s_addr=htonl(INADDR_LOOPBACK);addr.sin_port=htons(52381);
            if(bind(listener,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))!=0) {
                addr.sin_port=0;
                if(bind(listener,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))!=0) {close_socket(listener);listener=bad_socket;}
            }
            if(listener!=bad_socket) {
                if(listen(listener,4)==0) {
#ifdef _WIN32
                    int n=sizeof(addr);
#else
                    socklen_t n=sizeof(addr);
#endif
                    getsockname(listener,reinterpret_cast<sockaddr*>(&addr),&n);port=ntohs(addr.sin_port);
                } else {close_socket(listener);listener=bad_socket;}
            }
        }
        loading=true;
        worker=std::thread([this]{work();});
        if(listener!=bad_socket) server=std::thread([this]{serve();});
    }
    ~cleanup_engine() {
        stop=true; wake.notify_all();
        if(server.joinable()) server.join();
        if(listener!=bad_socket) close_socket(listener);
        if(worker.joinable()) worker.join();
        active.reset();
    }
    void work() {
        while(!stop) {
            {std::unique_lock<std::mutex> lock(worker_mutex);wake.wait(lock,[this]{return stop||requested;});if(stop)break;requested=false;}
            loading=true;
            {std::lock_guard<std::mutex> lock(status_mutex);++attempt;success=false;error.clear();}
            try {
                auto next=compile_generation(filter_path,runtime_dir,channel_count,sample_rate,bandwidth.load());
                if(next->g.guide && bandwidth.load() > (next->g.guide_bins-1)*sample_rate/(2*(next->g.guide_fft-1)))
                    throw std::runtime_error("Registered guide does not cover receive bandwidth; enlarge GUIDE_BINS and Reload");
                std::unique_ptr<Generation> previous;
                {
                    std::lock_guard<std::mutex> lock(audio_mutex);
                    previous=std::move(active);active=std::move(next);
                    runtime_fault=false;
                    fault_bandwidth=0.;
                    std::lock_guard<std::mutex> status_lock(status_mutex);
                    visible_geometry=active->g;visible_hash=active->hash;++generation_id;success=true;error.clear();
                    visible_guide_lanes=active->channels.empty()?0:cleanup_guide_lane_count(active->channels.front()->guide_plan);
                    visible_diagnostic=active->diagnostic;
                    if(visible_diagnostic) visible_diagnostic->generation=generation_id;
                    viewer_error.clear();
                }
                if(visible_geometry.viewer && visible_geometry.viewer_autostart) {
                    std::string launch_error;
                    { std::lock_guard<std::mutex> lock(python_mutex); Gil gil;
                      Ref module(PyImport_ImportModule("viewer"));
                      Ref result(module.p?PyObject_CallMethod(module.p,"launch","iK",static_cast<int>(port),static_cast<unsigned long long>(generation_id)):nullptr);
                      if(!result.p) launch_error=python_error();
                    }
                    if(!launch_error.empty()){std::lock_guard<std::mutex> lock(status_mutex);viewer_error=launch_error;}
                }
                previous.reset(); // Never free Python code or plans on the audio thread.
            } catch(const std::exception& e) {
                std::lock_guard<std::mutex> lock(status_mutex);error=e.what();success=false;
            } catch(...) {
                std::lock_guard<std::mutex> lock(status_mutex);error="Unknown compiler failure";success=false;
            }
            loading=false;
        }
    }
    std::string status() {
        std::lock_guard<std::mutex> lock(status_mutex);
        const auto& g=visible_geometry;
        std::string fault_message="Native Filter callback failed; bypass active until successful Reload";
        if(runtime_fault && fault_bandwidth.load()>0.){
            std::ostringstream detail;
            detail<<"Filter rejected "<<fault_bandwidth.load()<<" Hz receive bandwidth; registered guide covers "
                  <<(g.guide_bins-1)*sample_rate/(2*(g.guide_fft-1))
                  <<" Hz. Enlarge GUIDE_BINS or reduce bandwidth, then Reload; fault bypass is active";
            fault_message=detail.str();
        }
        std::ostringstream s;
        s<<"{\"abi\":1,\"generation\":"<<generation_id<<",\"attempt\":"<<attempt
         <<",\"loading\":"<<(loading?"true":"false")<<",\"success\":"<<(success&&!runtime_fault?"true":"false")
         <<",\"error\":"<<quoted(runtime_fault?fault_message:error)
         <<",\"bypass\":"<<(bypass?"true":"false")<<",\"squelch\":"<<(squelch?"true":"false")
         <<",\"harmonics\":"<<(harmonics?"true":"false")
         <<",\"runtime_fault\":"<<(runtime_fault?"true":"false")<<",\"port\":"<<port
         <<",\"sample_rate\":"<<sample_rate<<",\"channels\":"<<channel_count<<",\"bandwidth\":"<<bandwidth.load()
         <<",\"fft\":"<<g.fft<<",\"hop\":"<<g.hop<<",\"block\":"<<g.block<<",\"frames\":"<<g.frames
         <<",\"bins\":"<<g.bins<<",\"transform\":"<<quoted(g.transform?"odft":"rfft")
         <<",\"mask_source\":"<<quoted(g.mask_source_content?"content":"analysis")
         <<",\"analysis_mode\":"<<quoted(g.guide?"registered":"baseline")
         <<",\"guide_threads\":"<<visible_guide_lanes
         <<",\"guide_registration\":"<<(g.guide_registration?"true":"false")
         <<",\"guide_fft\":"<<g.guide_fft<<",\"guide_bins\":"<<g.guide_bins<<",\"guide_frames\":"<<g.guide_frames
         <<",\"guide_hop\":"<<g.guide_hop<<",\"guide_coverage_hz\":"<<(g.guide?(g.guide_bins-1)*sample_rate/(2*(g.guide_fft-1)):0.)
         <<",\"viewer\":"<<(g.viewer?"true":"false")<<",\"viewer_error\":"<<quoted(viewer_error)
         <<",\"latency_samples\":"<<(g.block?g.contexts*g.block-g.first*g.hop+g.fft/2:0)
         <<",\"blocks_processed\":"<<processed.load()<<",\"faults\":"<<faults.load()
         <<",\"last_block_ns\":"<<last_ns.load()<<",\"peak_block_ns\":"<<peak_ns.load()
         <<",\"stage_ns\":{\"guide\":"<<guide_ns.load()<<",\"filter\":"<<filter_ns.load()
         <<",\"observations\":"<<observation_ns.load()<<",\"views\":"<<view_ns.load()<<",\"registration\":"<<registration_ns.load()
         <<",\"stft\":"<<stft_ns.load()<<",\"publication\":"<<publication_ns.load()<<"}"
         <<",\"last_calls\":{\"filter\":"<<filter_calls.load()<<",\"guide\":"<<guide_calls.load()<<",\"shared_channels\":"<<shared_channels.load()<<"}"
         <<",\"sha256\":"<<quoted(visible_hash)<<"}";
        return s.str();
    }
    std::string command(const std::string& cmd) {
        if(cmd=="status") return status();
        if(cmd=="diagnostic") return diagnostic_json();
        if(cmd=="reload") {cleanup_reload(this);return status();}
        if(cmd=="bypass on") bypass=true;
        else if(cmd=="bypass off") bypass=false;
        else if(cmd=="bypass toggle") bypass=!bypass;
        else if(cmd=="squelch on") squelch=true;
        else if(cmd=="squelch off") squelch=false;
        else if(cmd=="squelch toggle") squelch=!squelch;
        else if(cmd=="harmonics on") harmonics=true;
        else if(cmd=="harmonics off") harmonics=false;
        else if(cmd=="harmonics toggle") harmonics=!harmonics;
        else return "{\"error\":\"commands: status, reload, bypass on|off|toggle, squelch on|off|toggle, harmonics on|off|toggle\"}";
        return status();
    }
    std::string diagnostic_json() {
        std::shared_ptr<Diagnostic> d;
        uint64_t current_generation;
        {std::lock_guard<std::mutex> lock(status_mutex);d=visible_diagnostic;current_generation=generation_id;}
        if(!d) return "{\"enabled\":false,\"generation\":"+std::to_string(current_generation)+"}";
        // Serialization runs on the control server. Audio uses try_lock and
        // simply drops a display publication while this snapshot is read.
        std::lock_guard<std::mutex> lock(d->mutex);
        const auto& g=d->g;
        std::ostringstream s;
        s.precision(9); // Preserve published float32 values across diagnostic JSON.

        s<<"{\"enabled\":true,\"generation\":"<<d->generation<<",\"sequence\":"<<d->sequence
         <<",\"channel\":0,\"sample_rate\":"<<sample_rate<<",\"block\":"<<g.block
         <<",\"bins\":"<<g.bins<<",\"emit\":"<<g.emit<<",\"hop\":"<<g.hop<<",\"fft\":"<<g.fft
         <<",\"half_bin\":"<<g.transform<<",\"guide_bins\":"<<g.view_bins<<",\"guide_emit\":"<<g.view_emit
         <<",\"guide_hop\":"<<g.view_hop<<",\"guide_denominator\":"<<g.view_frequency_denominator
         <<",\"guide_half_bin\":"<<g.view_half_bin
         <<",\"guide_center_offset\":"<<(static_cast<int64_t>(g.view_first*g.view_hop)-static_cast<int64_t>(g.first*g.hop))
         <<",\"action\":"<<d->action<<",\"bypass\":"<<(d->bypass?"true":"false")
         <<",\"analysis_paused\":"<<((d->bypass&&g.view_custom)?"true":"false")
         <<",\"squelch\":"<<(d->squelch?"true":"false")<<",\"gate_closed\":"<<(d->gate_closed?"true":"false")
         <<",\"valid\":"<<(d->valid?"true":"false");
        auto values=[&](const char* key,const std::vector<float>& data){s<<",\""<<key<<"\":[";bool first=true;for(float v:data){if(!first)s<<',';first=false;s<<v;}s<<']';};
        if(g.guide){s<<",\"evidence\":{\"reference_count\":"<<d->evidence[0]<<",\"reference_available\":"<<d->evidence[1]
            <<",\"reference_longest\":"<<d->evidence[2]<<",\"reference_run_available\":"<<d->evidence[3]
            <<",\"guide_count\":"<<d->evidence[4]<<",\"guide_available\":"<<d->evidence[5]
            <<",\"guide_longest\":"<<d->evidence[6]<<",\"guide_run_available\":"<<d->evidence[7]<<"}";}
        s<<",\"mask_stats\":[";
        for(int i=0;i<8;++i){if(i)s<<',';s<<d->mask_stats[i];}
        s<<"],\"crossing_stats\":[";
        for(int i=0;i<4;++i){if(i)s<<',';s<<d->crossing_stats[i];}
        s<<"],\"harmonic_edge_hz\":"<<d->harmonic_edge;
        if(g.surface){values("surface_mean",d->surface_mean);values("surface_variance",d->surface_variance);
            values("surface_floor",d->surface_floor);values("surface_mask",d->surface_mask);}
        if(g.surface_extra){values("surface_low",d->surface_low);values("surface_occupancy",d->surface_occupancy);values("surface_reference",d->surface_reference);}
        if(g.surface_shape)values("surface_shape",d->surface_shape);
        s<<",\"mask_stages\":[";for(int i=0;i<32;++i){if(i)s<<',';s<<d->mask_stages[i];}s<<']';
        s<<",\"population_stats\":[";for(int i=0;i<4;++i){if(i)s<<',';s<<d->population_stats[i];}s<<']';
        s<<",\"harmonic_stats\":[";for(int i=0;i<4;++i){if(i)s<<',';s<<d->harmonic_stats[i];}s<<']';
        values("guide",d->guide);values("input",d->input);values("filtered",d->filtered);s<<'}';return s.str();
    }
    void serve() {
        while(!stop) {
            fd_set set;FD_ZERO(&set);FD_SET(listener,&set);timeval tv{0,200000};
            if(select(static_cast<int>(listener)+1,&set,nullptr,nullptr,&tv)<=0) continue;
            socket_t client=accept(listener,nullptr,nullptr);if(client==bad_socket)continue;
            socket_timeout(client);
            std::string line;char buf[256];
            while(line.size()<4096 && !stop) {
                int n=static_cast<int>(recv(client,buf,sizeof(buf),0));if(n<=0)break;
                line.append(buf,n);if(line.find('\n')!=std::string::npos)break;
            }
            size_t end=line.find('\n');
            if(end!=std::string::npos && end<4096) {
                line.resize(end);if(!line.empty()&&line.back()=='\r')line.pop_back();
                send_text(client,command(line)+"\n");
            }
            close_socket(client);
        }
    }
    void block(Generation& gen) {
        const auto begin=std::chrono::steady_clock::now();
        const auto& g=gen.g;
        const bool dry_mode=bypass||runtime_fault;
        double config[5]={sample_rate,bandwidth.load(),squelch?1.:0.,harmonics?1.:0.,
            static_cast<double>(gen.block_index)*g.block-static_cast<double>((g.contexts-1)*g.block)};
        ++gen.block_index;
        bool failed=false;
        uint64_t guide_elapsed=0,filter_elapsed=0;
        uint64_t detail_elapsed[3]{};
        uint64_t stft_elapsed=0,publication_elapsed=0;
        uint32_t called_filter=0,called_guide=0,shared=0;
        int first_action=0;
        // Materialize the previous shared script state BEFORE channel zero
        // advances it. Never merge states again after distinct input histories.
        // Each inverse plan still runs every block to preserve its OLA tail.
        if(g.share_identical)for(uint32_t c=1;c<channel_count;++c){
            auto& ch=*gen.channels[c];const auto& first=*gen.channels[0];
            if(ch.state_mirrors_first &&
               (std::memcmp(ch.incoming_analysis.data(),first.incoming_analysis.data(),g.block*sizeof(double))!=0 ||
                std::memcmp(ch.incoming_content.data(),first.incoming_content.data(),g.block*sizeof(double))!=0)){
                std::copy(first.state.begin(),first.state.end(),ch.state.begin());
                ch.state_mirrors_first=false;
            }
        }
        for(uint32_t c=0;c<channel_count;++c) {
            auto& ch=*gen.channels[c];
            auto shift=[&](std::vector<double>& history,const std::vector<double>& in) {
                std::move(history.begin()+g.block,history.end(),history.begin());
                std::copy(in.begin(),in.end(),history.end()-g.block);
            };
            shift(ch.analysis_history,ch.incoming_analysis);shift(ch.content_history,ch.incoming_content);
            const size_t start=g.first*g.hop-g.fft/2;
            std::copy_n(ch.content_history.begin()+start,g.block,ch.dry.begin());
            auto forward=[&](std::vector<double>& history,std::vector<double>& spectrum) {
                auto start=std::chrono::steady_clock::now();
                check(bfft_stft_forward(ch.forward.p,history.data(),ch.packed.data()));
                for(size_t b=0;b<g.bins;++b)for(size_t t=0;t<g.frames;++t) {
                    const auto& v=ch.packed[b*g.frames+t];size_t i=(t*g.bins+b)*2;spectrum[i]=v.re;spectrum[i+1]=v.im;
                }
                stft_elapsed+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
            };
            int action=0;
            if(c>0 && g.share_identical && ch.state_mirrors_first){
                ++shared;
                const auto& first=*gen.channels[0];
                std::copy(first.content.begin(),first.content.end(),ch.content.begin());
                action=first_action;
            }else{
            forward(ch.content_history,ch.content);
            // Same transform and frame centers as final output; allocated at
            // generation construction, not inside the audio callback.
            if(c==0 && gen.diagnostic)for(size_t t=0;t<g.emit;++t)for(size_t b=0;b<g.bins;++b){
                const size_t i=((t+g.first)*g.bins+b)*2;
                ch.input_magnitude[t*g.bins+b]=static_cast<float>(std::hypot(ch.content[i],ch.content[i+1]));
            }
            if(!dry_mode || gen.diagnostic) forward(ch.analysis_history,ch.analysis);
            if(g.guide && !dry_mode) {
                const auto guide_begin=std::chrono::steady_clock::now();
                const auto& first_channel=*gen.channels[0];
                // Squelch evidence stays on early input. Mask observations use
                // the script-selected stream; reuse requires equality of that
                // stream, independently of the reference evidence source.
                const bool same_reference=c>0 && std::memcmp(ch.analysis_history.data(),first_channel.analysis_history.data(),
                                                       ch.analysis_history.size()*sizeof(double))==0;
                if(same_reference) std::copy(first_channel.reference_packed.begin(),first_channel.reference_packed.end(),ch.reference_packed.begin());
                else check(bfft_stft_forward(ch.reference_forward.p,ch.analysis_history.data(),ch.reference_packed.data()));
                const auto& mask_history=g.mask_source_content?ch.content_history:ch.analysis_history;
                const auto& first_history=g.mask_source_content?first_channel.content_history:first_channel.analysis_history;
                const bool same_guide=c>0 && std::memcmp(mask_history.data(),first_history.data(),mask_history.size()*sizeof(double))==0;
                if(same_guide) std::copy(first_channel.guide_cache.begin(),first_channel.guide_cache.end(),ch.guide_cache.begin());
                else {
                    ++called_guide;
                    if(cleanup_guide_run_stream(ch.guide_plan,mask_history.data(),mask_history.size(),g.block,ch.guide_cache.data()))failed=true;
                    uint64_t measured[3]{};cleanup_guide_timings(ch.guide_plan,measured);
                    for(int i=0;i<3;++i)detail_elapsed[i]+=measured[i];
                }
                std::copy(ch.guide_cache.begin(),ch.guide_cache.end(),ch.state.begin()+g.guide_slot*g.storage+g.guide_offset);
                for(size_t t=0;t<192;++t)for(size_t b=0;b<257;++b){const auto& v=ch.reference_packed[b*192+t];
                    ch.state[3*g.storage+64+t*257+b]=std::hypot(v.re,v.im);}
                guide_elapsed+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-guide_begin).count();
            }
            if(!dry_mode && !failed) {
                ++called_filter;
                const auto filter_begin=std::chrono::steady_clock::now();
                if(g.audio_history){
                    const auto& history=g.mask_source_content?ch.content_history:ch.analysis_history;
                    std::copy(history.begin(),history.end(),ch.state.begin()+g.audio_slot*g.storage+g.audio_offset);
                }
                action=gen.callback(ch.analysis.data(),ch.content.data(),ch.state.data(),config);
                filter_elapsed+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-filter_begin).count();
                if(action<0||action>2) {
                    if(g.guide && config[1]>(g.guide_bins-1)*sample_rate/(2*(g.guide_fft-1)))
                        fault_bandwidth=config[1];
                    failed=true;action=0;
                }
                if(action==1 && !std::all_of(ch.content.begin(),ch.content.end(),[](double v){return std::isfinite(v);})) {failed=true;action=0;}
            }
            // Refresh unmodified spectrum for passthrough even if a script mutated it.
            if(action==0 && !dry_mode) forward(ch.content_history,ch.content);
            }
            if(c==0)first_action=action;
            const auto inverse_begin=std::chrono::steady_clock::now();
            for(size_t b=0;b<g.bins;++b)for(size_t t=0;t<g.emit;++t) {
                size_t i=((t+g.first)*g.bins+b)*2;
                ch.emitted[b*g.emit+t]=action==2?bfft_complex{0.,0.}:bfft_complex{ch.content[i],ch.content[i+1]};
            }
            check(bfft_stft_inverse(ch.inverse.p,ch.emitted.data(),ch.rendered.data()));
            stft_elapsed+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-inverse_begin).count();
            if(c==0 && gen.diagnostic) {
                const auto publication_begin=std::chrono::steady_clock::now();
                auto& d=*gen.diagnostic;
                std::unique_lock<std::mutex> lock(d.mutex,std::try_to_lock);
                if(lock.owns_lock()) {
                    d.valid=true;
                    std::copy(ch.input_magnitude.begin(),ch.input_magnitude.end(),d.input.begin());
                    if(g.guide)std::copy_n(ch.state.data()+8,8,d.evidence);
                    if(g.slots && g.storage>=24)std::copy_n(ch.state.data()+16,8,d.mask_stats);
                    if(g.slots && g.storage>=28)std::copy_n(ch.state.data()+24,4,d.crossing_stats);
                    d.harmonic_edge=g.slots>1 && g.storage>4 ? ch.state[g.storage+4] : 0.;
                    if(g.storage>=64)std::copy_n(ch.state.data()+32,32,d.mask_stages);
                    if(g.slots && g.storage>=32)std::copy_n(ch.state.data()+28,4,d.population_stats);
                    if(g.slots>1 && g.storage>9)std::copy_n(ch.state.data()+g.storage+5,4,d.harmonic_stats);
                    for(size_t t=0;t<g.view_emit;++t)for(size_t b=0;b<g.view_bins;++b){
                        double v;
                        if(dry_mode && g.view_custom) v=0.; // explicit paused analysis, never stale frames
                        else if(g.view_custom) v=ch.state[g.view_slot*g.storage+g.view_offset+(g.view_first+t)*g.view_bins+b];
                        else {size_t i=((g.first+t)*g.bins+b)*2;v=std::hypot(ch.analysis[i],ch.analysis[i+1]);}
                        if(!std::isfinite(v)||v<0||v>3.402823466e38){v=0.;d.valid=false;}
                        d.guide[t*g.view_bins+b]=static_cast<float>(v);
                    }
                    if(g.surface) {
                        const size_t plane=g.surface_frames*g.surface_bins;
                        for(size_t t=0;t<g.view_emit;++t)for(size_t b=0;b<g.view_bins;++b) {
                            const size_t row=(g.view_first+t)*g.view_hop/g.surface_hop;
                            const size_t src=row*g.surface_bins+b,dst=t*g.view_bins+b;
                            auto copy_surface=[&](std::vector<float>& out,size_t offset){
                                double v=ch.state[offset+src];
                                if(!std::isfinite(v)||v<0||v>3.402823466e38){v=0.;d.valid=false;}
                                out[dst]=static_cast<float>(v);
                            };
                            copy_surface(d.surface_mean,g.surface_offset);
                            copy_surface(d.surface_variance,g.surface_offset+plane);
                            copy_surface(d.surface_floor,g.surface_offset+2*plane);
                            copy_surface(d.surface_mask,g.surface_mask_offset);
                            if(g.surface_shape)copy_surface(d.surface_shape,g.surface_shape_offset);
                            if(g.surface_extra){copy_surface(d.surface_low,g.surface_low_offset);copy_surface(d.surface_occupancy,g.surface_occupancy_offset);copy_surface(d.surface_reference,g.surface_reference_offset);}
                        }
                    }
                    for(size_t t=0;t<g.emit;++t)for(size_t b=0;b<g.bins;++b){
                        const auto& v=ch.emitted[b*g.emit+t];
                        double magnitude=std::hypot(v.re,v.im);
                        if(!std::isfinite(magnitude)||magnitude>3.402823466e38){magnitude=0.;d.valid=false;}
                        d.filtered[t*g.bins+b]=static_cast<float>(magnitude);
                    }
                    d.action=action;d.bypass=dry_mode;d.squelch=config[2]!=0.;
                    d.gate_closed=d.squelch && action==2 && !dry_mode;
                    d.sequence=processed.load()+1;
                }
                publication_elapsed+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-publication_begin).count();
            }
            // SKIP adds only previous windowed overlap, naturally tapering to zero.
            const auto& source=action==0?ch.dry:ch.rendered;
            for(size_t i=0;i<g.block;++i) {
                if(!std::isfinite(source[i]) || std::abs(source[i])>3.402823466e38) failed=true;
                gen.output[i*channel_count+c]=static_cast<float>(source[i]);
            }
        }
        if(failed) {
            runtime_fault=true;++faults;
            for(uint32_t c=0;c<channel_count;++c) for(size_t i=0;i<g.block;++i)
                gen.output[i*channel_count+c]=static_cast<float>(gen.channels[c]->dry[i]);
            if(gen.diagnostic){
                auto& d=*gen.diagnostic;std::unique_lock<std::mutex> lock(d.mutex,std::try_to_lock);
                if(lock.owns_lock()){d.action=0;d.bypass=true;d.gate_closed=false;d.valid=false;}
            }
        }
        gen.output_pos=0;gen.output_count=g.block;++processed;
        uint64_t ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();
        last_ns=ns;uint64_t peak=peak_ns.load();while(peak<ns&&!peak_ns.compare_exchange_weak(peak,ns)){}
        guide_ns=guide_elapsed;filter_ns=filter_elapsed;
        observation_ns=detail_elapsed[0];view_ns=detail_elapsed[1];registration_ns=detail_elapsed[2];
        stft_ns=stft_elapsed;publication_ns=publication_elapsed;
        filter_calls=called_filter;guide_calls=called_guide;shared_channels=shared;
    }
};

extern "C" {
uint32_t cleanup_abi_version(){return 1;}
cleanup_engine* cleanup_create(const char* path,const char* runtime,double sr,uint32_t channels) {
    if(!path||!runtime||!std::isfinite(sr)||sr<8000.||sr>384000.||channels<1||channels>2)return nullptr;
    try{return new cleanup_engine(path,runtime,sr,channels);}catch(...){return nullptr;}
}
void cleanup_destroy(cleanup_engine* e){delete e;}
int cleanup_reload(cleanup_engine* e) {
    if(!e)return -1;
    std::lock_guard<std::mutex> lock(e->worker_mutex);
    if(e->loading||e->requested)return 0;
    e->requested=true;e->loading=true;e->wake.notify_one();return 1;
}
void cleanup_set_bypass(cleanup_engine* e,int v){if(e)e->bypass=v!=0;}
void cleanup_set_squelch(cleanup_engine* e,int v){if(e)e->squelch=v!=0;}
void cleanup_set_harmonics(cleanup_engine* e,int v){if(e)e->harmonics=v!=0;}
void cleanup_set_bandwidth(cleanup_engine* e,double hz){if(e&&std::isfinite(hz)&&hz>0)e->bandwidth=std::min(hz,e->sample_rate*.5);}
uint16_t cleanup_control_port(cleanup_engine* e){return e?e->port:0;}
uint32_t cleanup_status(cleanup_engine* e,char* dest,uint32_t capacity) {
    if(!e)return 0;
    try {auto s=e->status();if(dest&&capacity){size_t n=std::min<size_t>(s.size(),capacity-1);memcpy(dest,s.data(),n);dest[n]=0;}return static_cast<uint32_t>(s.size()+1);}catch(...){return 0;}
}
int cleanup_process_pair(cleanup_engine* e,const float* analysis,const float* content,float* output,uint32_t frames) {
    if(!e||!analysis||!content||!output)return -1;
    std::lock_guard<std::mutex> lock(e->audio_mutex);
    if(!e->active) {std::memmove(output,content,static_cast<size_t>(frames)*e->channel_count*sizeof(float));return 0;}
    auto& g=*e->active;
    try {
        for(uint32_t i=0;i<frames;++i) {
            // Read inputs before writing outputs, including the in-place case.
            for(uint32_t c=0;c<e->channel_count;++c) {
                double a=analysis[i*e->channel_count+c],x=content[i*e->channel_count+c];
                g.channels[c]->incoming_analysis[g.input_pos]=std::isfinite(a)?a:0.;
                g.channels[c]->incoming_content[g.input_pos]=std::isfinite(x)?x:0.;
            }
            for(uint32_t c=0;c<e->channel_count;++c)
                output[i*e->channel_count+c]=g.output_count?g.output[g.output_pos*e->channel_count+c]:0.f;
            if(g.output_count) {++g.output_pos;--g.output_count;}
            if(++g.input_pos==g.g.block){g.input_pos=0;e->block(g);}
        }
        return e->runtime_fault?-2:0;
    } catch(...) {
        e->runtime_fault=true;++e->faults;
        // Already-emitted samples remain valid; fail closed for this exceptional callback.
        std::fill_n(output,static_cast<size_t>(frames)*e->channel_count,0.f);return -2;
    }
}
int cleanup_process(cleanup_engine* e,const float* input,float* output,uint32_t frames){return cleanup_process_pair(e,input,input,output,frames);}
}
