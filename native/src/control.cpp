#include "socket.hpp"
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <string>
int main(int argc,char** argv) {
    int port=52381,index=1;
    if(argc>2&&std::string(argv[1])=="--port") {port=std::atoi(argv[2]);index=3;}
    if(port<1||port>65535){std::cerr<<"Invalid port\n";return 2;}
    std::string command;
    for(int i=index;i<argc;++i){if(!command.empty())command+=' ';command+=argv[i];}
    if(command.empty())command="status";
    init_sockets();auto s=socket(AF_INET,SOCK_STREAM,IPPROTO_TCP);if(s==bad_socket)return 1;
    socket_timeout(s);
    sockaddr_in addr{};addr.sin_family=AF_INET;addr.sin_port=htons(static_cast<uint16_t>(port));addr.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    if(connect(s,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))!=0){std::cerr<<"Cannot connect to Cleanup DLL control port "<<port<<"\n";close_socket(s);return 1;}
    if(!send_text(s,command+"\n")){close_socket(s);return 1;}
    std::string response;char buf[4096];
    while(response.size()<1024*1024){int n=static_cast<int>(recv(s,buf,sizeof(buf),0));if(n<=0)break;response.append(buf,n);if(response.find('\n')!=std::string::npos)break;}
    close_socket(s);std::cout<<response;
    return response.empty()?1:0;
}
