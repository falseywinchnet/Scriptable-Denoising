// Native Dear ImGui/ImPlot viewer. No Python, Direct3D, or host GUI dependency.
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
using Socket=SOCKET;
static void close_socket(Socket s){closesocket(s);}
#else
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
using Socket=int;
static void close_socket(Socket s){close(s);}
#endif
#include <GLFW/glfw3.h>
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif
#include <imgui.h>
#include <implot.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
using Json=nlohmann::json;
using Clock=std::chrono::steady_clock;
static constexpr float missing=std::numeric_limits<float>::quiet_NaN();
struct Connection {
    Socket fd;
    explicit Connection(int port){
        fd=socket(AF_INET,SOCK_STREAM,0);
        if(fd==Socket(-1))throw std::runtime_error("socket failed");
#ifdef _WIN32
        DWORD timeout=500;
        setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,reinterpret_cast<char*>(&timeout),sizeof(timeout));
        setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,reinterpret_cast<char*>(&timeout),sizeof(timeout));
#else
        timeval timeout{0,500000};
        setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
        setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));
#ifdef SO_NOSIGPIPE
        int one=1;setsockopt(fd,SOL_SOCKET,SO_NOSIGPIPE,&one,sizeof(one));
#endif
#endif
        sockaddr_in address{};address.sin_family=AF_INET;address.sin_port=htons(static_cast<unsigned short>(port));
        address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
        if(connect(fd,reinterpret_cast<sockaddr*>(&address),sizeof(address))!=0){close_socket(fd);throw std::runtime_error("diagnostic connection unavailable");}
    }
    ~Connection(){close_socket(fd);}
    Json read(){
#ifdef MSG_NOSIGNAL
        constexpr int flags=MSG_NOSIGNAL;
#else
        constexpr int flags=0;
#endif
        const char command[]="diagnostic\n";
        int sent=0;
        while(sent<11){int n=send(fd,command+sent,11-sent,flags);if(n<=0)throw std::runtime_error("diagnostic send failed");sent+=n;}
        std::string text;char buffer[65536];
        while(text.size()<64*1024*1024){
            int n=recv(fd,buffer,sizeof(buffer),0);
            if(n<=0)throw std::runtime_error("incomplete diagnostic response");
            text.append(buffer,n);
            auto end=text.find('\n');if(end!=std::string::npos)return Json::parse(text.begin(),text.begin()+end);
        }
        throw std::runtime_error("diagnostic response exceeds limit");
    }
};
struct Mailbox {std::mutex mutex;Json value;std::string error;};
struct Matrix {
    int width=0,bins=0;double hop=0,fs=0,denominator=0,half=0,offset=0;
    std::vector<float> values;
    void reset(int count,int frequencies,double rate,double step,double divisor,double halfbin,double origin,double seconds){
        if(count<1||frequencies<1||frequencies>16384||rate<=0||step<=0||divisor<=0)throw std::runtime_error("invalid diagnostic geometry");
        // Bound history storage even for pathological FFT/hop choices.
        width=std::max(1,std::min({4096,2097152/frequencies,int(std::ceil(seconds*rate/step))}));
        bins=frequencies;fs=rate;hop=step;denominator=divisor;half=halfbin;offset=origin;
        values.assign(size_t(width)*bins,missing);
    }
    void append(const std::vector<float>& incoming,int columns,uint64_t gaps){
        if(incoming.size()!=size_t(columns)*bins)throw std::runtime_error("diagnostic matrix size mismatch");
        int shift=int(std::min<uint64_t>(width,std::min<uint64_t>(gaps,width)*columns+columns));
        std::memmove(values.data(),values.data()+size_t(shift)*bins,size_t(width-shift)*bins*sizeof(float));
        std::fill(values.end()-size_t(shift)*bins,values.end(),missing);
        int keep=std::min(width,columns);
        std::copy(incoming.end()-size_t(keep)*bins,incoming.end(),values.end()-size_t(keep)*bins);
    }
};
struct History {
    uint64_t generation=0,sequence=0;Matrix guide,filtered;
    std::vector<double> gate,gate_x;Json last;
    bool accept(const Json& j,double seconds){
        if(!j.value("enabled",false)||j.value("sequence",uint64_t(0))==0)return false;
        auto gen=j.at("generation").get<uint64_t>(),seq=j.at("sequence").get<uint64_t>();
        if(gen==generation&&seq<=sequence)return false;
        if(gen!=generation){
            generation=gen;sequence=0;
            double fs=j.at("sample_rate");
            guide.reset(j.at("guide_emit"),j.at("guide_bins"),fs,j.at("guide_hop"),j.at("guide_denominator"),j.at("guide_half_bin"),j.at("guide_center_offset").get<double>()/fs,seconds);
            filtered.reset(j.at("emit"),j.at("bins"),fs,j.at("hop"),j.at("fft"),j.at("half_bin"),0,seconds);
            gate.assign(filtered.width,std::numeric_limits<double>::quiet_NaN());gate_x.resize(filtered.width);
            for(int i=0;i<filtered.width;++i)gate_x[i]=(i-filtered.width)*filtered.hop/fs;
        }
        uint64_t gaps=sequence?seq-sequence-1:0;
        guide.append(j.at("guide").get<std::vector<float>>(),j.at("guide_emit"),gaps);
        filtered.append(j.at("filtered").get<std::vector<float>>(),j.at("emit"),gaps);
        int emit=j.at("emit"),shift=int(std::min<uint64_t>(gate.size(),std::min<uint64_t>(gaps,gate.size())*emit+emit));
        std::memmove(gate.data(),gate.data()+shift,(gate.size()-shift)*sizeof(double));
        std::fill(gate.end()-shift,gate.end(),std::numeric_limits<double>::quiet_NaN());
        std::fill(gate.end()-std::min<int>(gate.size(),emit),gate.end(),j.value("gate_closed",false)?1.:0.);
        sequence=seq;last=j;return true;
    }
};
struct Texture {
    GLuint id=0;int width=0,height=0;double xmin=0,xmax=0,ymin=0,ymax=0;
    std::vector<unsigned char> pixels;
    ~Texture(){if(id)glDeleteTextures(1,&id);}
    void upload(const Matrix& m){
        static const float anchors[9][3]={{0,0,4},{31,12,72},{85,15,109},{136,34,106},{186,54,85},{227,89,51},{249,140,10},{249,201,50},{252,255,164}};
        GLint limit=0;glGetIntegerv(GL_MAX_TEXTURE_SIZE,&limit);
        if(m.width>limit||m.bins>limit)throw std::runtime_error("diagnostic image exceeds GPU texture dimensions");
        pixels.resize(size_t(m.width)*m.bins*4);
        for(int b=0;b<m.bins;++b)for(int t=0;t<m.width;++t){
            float v=m.values[size_t(t)*m.bins+b];size_t dest=(size_t(m.bins-1-b)*m.width+t)*4;
            float level=std::clamp((20.f*std::log10(std::max(v,1e-6f))+120.f)/140.f,0.f,1.f)*8;
            if(!std::isfinite(v)){pixels[dest]=20;pixels[dest+1]=20;pixels[dest+2]=23;}
            else {int a=std::min(7,int(level));float f=level-a;for(int c=0;c<3;++c)pixels[dest+c]=static_cast<unsigned char>(anchors[a][c]*(1-f)+anchors[a+1][c]*f);}
            pixels[dest+3]=255;
        }
        if(!id){glGenTextures(1,&id);glBindTexture(GL_TEXTURE_2D,id);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);}
        glBindTexture(GL_TEXTURE_2D,id);glPixelStorei(GL_UNPACK_ALIGNMENT,1);
        if(width!=m.width||height!=m.bins)glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,m.width,m.bins,0,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());
        else glTexSubImage2D(GL_TEXTURE_2D,0,0,0,m.width,m.bins,GL_RGBA,GL_UNSIGNED_BYTE,pixels.data());
        width=m.width;height=m.bins;xmin=-width*m.hop/m.fs+m.offset;xmax=m.offset;
        double df=m.fs/m.denominator;ymin=(.5*m.half-.5)*df;ymax=(height+.5*m.half-.5)*df;
    }
    void plot(const char* label,float height_pixels,double upper_hz){
        if(ImPlot::BeginPlot(label,ImVec2(-1,height_pixels))){
            ImPlot::SetupAxes("Seconds before latest block","Hz");
            ImPlot::SetupAxesLimits(xmin,xmax,0,upper_hz,ImGuiCond_Once);
            if(id)ImPlot::PlotImage("magnitude",static_cast<ImTextureID>(id),ImPlotPoint(xmin,ymin),ImPlotPoint(xmax,ymax));
            ImPlot::EndPlot();
        }
    }
};
static void result(const std::string& path,const Json& value){
    if(path.empty())return;
    std::ofstream stream(path+".tmp");stream<<value.dump();stream.close();
    std::filesystem::remove(path);
    std::filesystem::rename(path+".tmp",path);
}
int main(int argc,char** argv){
    int port=52381;uint64_t expected=0;double seconds=6,exit_after=0;std::string ready_file,report_file,capture_file;
    bool self_test=false;
    for(int i=1;i<argc;++i){std::string arg=argv[i];
        if(arg=="--self-test"){self_test=true;continue;}
        if(i+1>=argc){std::cerr<<"Missing option value\n";return 2;}
        std::string value=argv[++i];
        if(arg=="--port")port=std::stoi(value);else if(arg=="--generation")expected=std::stoull(value);
        else if(arg=="--seconds")seconds=std::stod(value);else if(arg=="--exit-after")exit_after=std::stod(value);
        else if(arg=="--ready-file")ready_file=value;else if(arg=="--report")report_file=value;
        else if(arg=="--capture")capture_file=value;else {std::cerr<<"Unknown option: "<<arg<<'\n';return 2;}}
    if(port<1||port>65535||seconds<1||seconds>30)return 2;
#ifdef _WIN32
    WSADATA winsock; if(WSAStartup(MAKEWORD(2,2),&winsock))return 2;
#endif
    glfwSetErrorCallback([](int code,const char* text){std::cerr<<"GLFW "<<code<<": "<<text<<'\n';});
    if(!glfwInit())return 3;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,2);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT,GL_TRUE);
    GLFWwindow* window=glfwCreateWindow(1180,840,"Cleanup - native diagnostics",nullptr,nullptr);
    const char* shader="#version 150";
    if(!window){
        // Compatibility drivers (including this Wine setup) may expose only GL
        // 2.1. ImGui's programmable OpenGL backend supports GLSL 120 as well.
        glfwDefaultWindowHints();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,2);glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,1);
        window=glfwCreateWindow(1180,840,"Cleanup - native diagnostics",nullptr,nullptr);
        shader="#version 120";
    }
    if(!window){glfwTerminate();return 3;}
    glfwMakeContextCurrent(window);glfwSwapInterval(1);
    std::string renderer=reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    std::string version=reinterpret_cast<const char*>(glGetString(GL_VERSION));
    std::cerr<<"OpenGL "<<version<<" | "<<renderer<<'\n';
    IMGUI_CHECKVERSION();ImGui::CreateContext();ImPlot::CreateContext();ImGui::StyleColorsDark();
    ImGui::GetIO().IniFilename=nullptr;
    ImGui_ImplGlfw_InitForOpenGL(window,true);
    if(!ImGui_ImplOpenGL3_Init(shader)){glfwDestroyWindow(window);glfwTerminate();return 3;}
    Mailbox mailbox;std::atomic<bool> stop{false};
    std::thread worker([&]{int failures=0;while(!stop){
        try{auto j=Connection(port).read();failures=0;
            if(expected&&(j.value("generation",uint64_t(0))!=expected||!j.value("enabled",false)))j=Json{{"exit",true}};
            std::lock_guard<std::mutex> lock(mailbox.mutex);mailbox.value=std::move(j);mailbox.error.clear();
        }catch(const std::exception& e){std::lock_guard<std::mutex> lock(mailbox.mutex);mailbox.error=e.what();if(expected&&++failures>=12)mailbox.value=Json{{"exit",true}};}
        for(int i=0;i<8&&!stop;++i)std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }});
    int code=0;uint64_t rendered=0,displayed=0;bool pause=false,dirty=false,ready=false;auto start=Clock::now();
    History history;Json audit={{"renderer",renderer},{"opengl",version},{"glsl",shader},
        {"core_profile",glfwGetWindowAttrib(window,GLFW_OPENGL_PROFILE)==GLFW_OPENGL_CORE_PROFILE},
        {"forward_compatible",bool(glfwGetWindowAttrib(window,GLFW_OPENGL_FORWARD_COMPAT))},{"success",false}};
    try{
        Texture guide,filtered;std::vector<double> gate,gate_x;Json shown;
        int test_stage=0;double test_start=0;uint64_t frozen_seq=0;std::vector<unsigned char> frozen;
        while(!glfwWindowShouldClose(window)){
            double elapsed=std::chrono::duration<double>(Clock::now()-start).count();
            if(exit_after&&elapsed>=exit_after)break;
            Json item;std::string error;{std::lock_guard<std::mutex> lock(mailbox.mutex);item=std::move(mailbox.value);mailbox.value=nullptr;error=mailbox.error;}
            if(item.is_object()&&item.value("exit",false))break;
            if(item.is_object()&&history.accept(item,seconds))dirty=true;
            glfwPollEvents();ImGui_ImplOpenGL3_NewFrame();ImGui_ImplGlfw_NewFrame();ImGui::NewFrame();
            ImGui::SetNextWindowPos(ImVec2(0,0));ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
            ImGui::Begin("Cleanup diagnostics",nullptr,ImGuiWindowFlags_NoDecoration|ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoSavedSettings);
            auto toggle=[&]{pause=!pause;dirty=true;};
            if(ImGui::Button(pause?"Resume":"Pause"))toggle();ImGui::SameLine();
            ImGui::TextUnformatted(pause?"Paused - filtering and polling continue":"Live - channel 1");
            if(self_test&&history.sequence){
                if(test_stage==0){test_start=elapsed;test_stage=1;}
                else if(test_stage==1&&elapsed-test_start>1&&displayed){toggle();frozen=filtered.pixels;frozen_seq=displayed;test_stage=2;}
                else if(test_stage==2&&elapsed-test_start>3){
                    if(filtered.pixels!=frozen||displayed!=frozen_seq||history.sequence<=frozen_seq)throw std::runtime_error("pause test failed");
                    audit["pause_ingestion_continued"]=true;toggle();test_stage=3;
                }else if(test_stage==3&&elapsed-test_start>4){
                    if(displayed<=frozen_seq)throw std::runtime_error("resume test failed");
                    audit["pause_resume_passed"]=true;glfwSetWindowShouldClose(window,GL_TRUE);
                }
            }
            if(dirty&&!pause&&history.sequence){
                guide.upload(history.guide);filtered.upload(history.filtered);gate=history.gate;gate_x=history.gate_x;
                shown=history.last;displayed=history.sequence;dirty=false;
            }
            if(displayed&&shown.value("source",std::string())=="offline-replay")ImGui::TextUnformatted("OFFLINE REPLAY - experimental guide mask");
            if(displayed)ImGui::Text("Generation %llu | block %llu | %s | %s",static_cast<unsigned long long>(shown.at("generation").get<uint64_t>()),static_cast<unsigned long long>(displayed),shown.value("bypass",false)?"BYPASS":shown.value("gate_closed",false)?"SQUELCHED":"OPEN",shown.value("valid",false)?"valid":"invalid diagnostic data");
            else ImGui::TextUnformatted("Waiting for filter data");
            if(displayed&&shown.at("guide_denominator")!=shown.at("fft"))ImGui::TextUnformatted("Different transforms: raw magnitude colors are not power-calibrated across panels");
            if(displayed&&shown.value("analysis_paused",false))ImGui::TextUnformatted("Bypass: expensive analysis is paused; new upper-panel columns are blank");
            if(!error.empty())ImGui::TextUnformatted(error.c_str());
            ImGui::TextUnformatted("Inferno | -120 to +20 dB | top: analysis | bottom: final coefficients before inversion");
            if(ImPlot::BeginPlot("Squelch: 1 closed, 0 open",ImVec2(-1,95),ImPlotFlags_NoLegend)){
                ImPlot::SetupAxes("Seconds before latest block","Gate");
                ImPlot::SetupAxesLimits(-seconds,0,-.1,1.1,ImGuiCond_Always);
                if(!gate.empty())ImPlot::PlotStairs("gate",gate_x.data(),gate.data(),int(gate.size()));ImPlot::EndPlot();
            }
            float panel=std::max(120.f,(ImGui::GetContentRegionAvail().y-12)*.5f);
            if(displayed){guide.plot("Analysis - before filtering",panel,std::min(4000.,history.filtered.fs*.5));filtered.plot("Final result - immediately before inversion",panel,std::min(4000.,history.filtered.fs*.5));}
            ImGui::End();ImGui::Render();int width,height;glfwGetFramebufferSize(window,&width,&height);
            glViewport(0,0,width,height);glClearColor(.06f,.06f,.07f,1);glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            if(!capture_file.empty()&&displayed&&elapsed>2){
                std::vector<unsigned char> rgb(size_t(width)*height*3);glPixelStorei(GL_PACK_ALIGNMENT,1);glReadPixels(0,0,width,height,GL_RGB,GL_UNSIGNED_BYTE,rgb.data());
                std::ofstream f(capture_file,std::ios::binary);f<<"P6\n"<<width<<' '<<height<<"\n255\n";
                for(int y=height-1;y>=0;--y)f.write(reinterpret_cast<char*>(rgb.data()+size_t(y)*width*3),width*3);
                capture_file.clear();
            }
            glfwSwapBuffers(window);++rendered;
            if(!ready){result(ready_file,Json{{"success",true},{"renderer",renderer},{"opengl",version}});ready=true;}
        }
        if(self_test&&!audit.value("pause_resume_passed",false))throw std::runtime_error("self-test ended before completion");
        audit["success"]=true;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';audit["error"]=e.what();code=1;}
    stop=true;worker.join();
    audit["rendered_frames"]=rendered;audit["generation"]=history.generation;audit["received_sequence"]=history.sequence;audit["displayed_sequence"]=displayed;
    result(report_file,audit);std::cout<<audit.dump()<<'\n';
    ImGui_ImplOpenGL3_Shutdown();ImGui_ImplGlfw_Shutdown();ImPlot::DestroyContext();ImGui::DestroyContext();
    glfwDestroyWindow(window);glfwTerminate();
#ifdef _WIN32
    WSACleanup();
#endif
    return code;
}
